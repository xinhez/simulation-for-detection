#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import cv2
import glob
import os
import sys

from carlaModule import carla
from carlaSyncMode import CarlaSyncMode

import random
import pygame
import numpy as np
from datetime import datetime

# Hyperparameters Here
n_vehicle = 3 
n_walkers = 0
show_img = 0
show_recording = 1

# Camera position setting
camera_x = 5.5 
camera_y = 0
camera_z = 1

# Data description
town_num = 'town01'

target_color = 'Red'

step_size = 5
image_goal = 10000-6331

# Target threshold size
target_size = 'L'
if target_size == 'S':
    fire_hydrant_qualified_size_min = 10*10 # medium
    fire_hydrant_qualified_size_max = 32*32 # small
elif target_size == 'M':
    fire_hydrant_qualified_size_min = 32*32 # medium
    fire_hydrant_qualified_size_max = 96*96 # medium
elif target_size == 'L':
    fire_hydrant_qualified_size_min = 96*96 # large
    fire_hydrant_qualified_size_max = 1080*1920 # large

out_seg = '%s_%s_%s_seg' % (town_num, target_size, target_color)
out_rgb = '%s_%s_%s_rgb' % (town_num, target_size, target_color)

img_resolution = (1920, 1080)

# Camera angle setting
# if target_size == 'S':
#     camera_pitch = 0
#     camera_yaw = 15
# elif target_size == 'M':
#     camera_pitch = -15
#     camera_yaw = 300
# elif target_size == 'L':
camera_pitch = -15
camera_yaw = 90

# End of Hyperparameters

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def postprocess(seg_file, rgb_file):
    seg_img = cv2.imread(seg_file)
    rgb_img = cv2.imread(rgb_file)
    _, timg = cv2.threshold(seg_img[:, :, 2], 110, 220, cv2.THRESH_BINARY)
    output = cv2.connectedComponentsWithStats(timg, 8, cv2.CV_32S)

    num_labels = output[0]
    stats = output[2]

    fire_hydrant_found = 0
    fire_hydrant_qualified = 0
    for i in range(num_labels):
        if (
            stats[i, cv2.CC_STAT_AREA] > 0 and 
            stats[i, cv2.CC_STAT_WIDTH] != 1920 and stats[i, cv2.CC_STAT_HEIGHT] != 1080 and 
            stats[i, cv2.CC_STAT_HEIGHT] > 10 # Check for chains 
            ): 

            fire_hydrant_found += 1
            
            if fire_hydrant_qualified_size_min < stats[i, cv2.CC_STAT_AREA] < fire_hydrant_qualified_size_max :
                fire_hydrant_qualified += 1
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if show_img == 1:
                    cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (128, 255, 0))

    print('found', fire_hydrant_found, 'qualified', fire_hydrant_qualified)
    if show_img == 1 and fire_hydrant_found == 1:
        cv2.imshow('Boundig Boxes', rgb_img)
        cv2.waitKey(0)

    if fire_hydrant_found != 1 or fire_hydrant_qualified != 1:
        os.remove(seg_file)
        os.remove(rgb_file)
        return 0
    else:
        print('saved img', rgb_file)
        return 1

def main():
    # weather parameters 
    weather_parameters = {
        'cloudiness'             :   0, #   0 ~ 100
        'precipitation'          :   0, #   0 ~ 100
        'precipitation_deposits' :   0, #   0 ~ 100
        'wind_intensity'         :   0, #   0 ~ 100
        'sun_azimuth_angle'      :   0, #   0 ~ 360
        'sun_altitude_angle'     :  50, # -90 ~  90
        'fog_density'            :   0, #   0 ~ 180
        'fog_distance'           :   0, #   0 ~ infinite
        'wetness'                :   0, #   0 ~ 100
    }
    weather_keys = list(weather_parameters.keys())
    weather_index = 0
    
    image_collected = 0
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    try:
        os.mkdir(out_seg)
        os.mkdir(out_rgb)
    except:
        print("Did not create image directories")
    actor_list = []
    pygame.init()

    if show_recording == 1:
        display = pygame.display.set_mode(img_resolution,
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()

        ### Spawn vehicles
        spawn_points = m.get_spawn_points()
        print('%d spawn points generated' % len(spawn_points))
        random.shuffle(spawn_points)
        waypoints = []
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
            
        for i in range(n_vehicle):
            vehicle = world.spawn_actor(
                random.choice(vehicle_blueprints),
                spawn_points[i])
            actor_list.append(vehicle)
            vehicle.set_simulate_physics(False)
            waypoints.append(m.get_waypoint(spawn_points[i].location))

        ### Spawn Pedestrians
        walkers_list = []
        # # 0. Choose a blueprint fo the walkers
        blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprintsWalkers)

        # 1. Take all the random locations to spawn
        spawn_points = []
        for i in range(n_walkers):
            spawn_point = carla.Transform()
            spawn_point.location = world.get_random_location_from_navigation()
            spawn_points.append(spawn_point)

        # 2. Build the batch of commands to spawn the pedestrians
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # 2.1 apply the batch
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            walkers_list.append({"id": results[i].actor_id})

        # 3. Spawn walker AI controllers for each walker
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

        # 3.1 apply the batch
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            walkers_list[i]["con"] = results[i].actor_id

        # 4. Put altogether the walker and controller ids
        all_id = []
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)
        actor_list.extend(all_actors)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_actors), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # random max speed
            all_actors[i].set_max_speed(1 + random.random())  

        camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        camera_rgb_bp.set_attribute('image_size_x', '%d'%img_resolution[0])
        camera_rgb_bp.set_attribute('image_size_y', '%d'%img_resolution[1])
        camera_rgb = world.spawn_actor(
            camera_rgb_bp,
            carla.Transform(carla.Location(x=camera_x, y=camera_y, z=camera_z), carla.Rotation(pitch=camera_pitch, yaw=camera_yaw)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_semseg_bp.set_attribute('image_size_x', '%d'%img_resolution[0])
        camera_semseg_bp.set_attribute('image_size_y', '%d'%img_resolution[1])
        camera_semseg = world.spawn_actor(
            camera_semseg_bp,
            carla.Transform(carla.Location(x=camera_x, y=camera_y, z=camera_z), carla.Rotation(pitch=camera_pitch, yaw=camera_yaw)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)
        clock_count = 1

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=10) as sync_mode:
            while True:
                # Begin change weather
                weather = carla.WeatherParameters(**weather_parameters)
                world.set_weather(weather)

                # weather_parameters[weather_keys[weather_index]] += 25
                # if weather_keys[weather_index] == 'sun_azimuth_angle':
                #     weather_parameters[weather_keys[weather_index]] %= 360
                # elif weather_keys[weather_index] == 'sun_altitude_angle':
                #     weather_parameters[weather_keys[weather_index]] += 90
                #     weather_parameters[weather_keys[weather_index]] %= 180
                #     weather_parameters[weather_keys[weather_index]] -= 90
                # elif weather_keys[weather_index] == 'fog_density':
                #     weather_parameters[weather_keys[weather_index]] %= 180
                # else:
                #     weather_parameters[weather_keys[weather_index]] %= 100
                # weather_index += 1
                # weather_index %= len(weather_keys)
                # End change weather

                if should_quit(): return
                if image_collected == image_goal: return
                clock.tick()
                clock_count += 1

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                for i in range(n_vehicle):
                    waypoints[i] = random.choice(waypoints[i].next(1.5))
                    actor_list[i].set_transform(waypoints[i].transform)

                # if False:
                if clock_count % step_size == 0:
                    clock_count = 1
                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)

                    seg_file = '%s/%s_%06d.png' % (out_seg, current_time, image_semseg.frame)
                    rgb_file = '%s/%s_%06d.png' % (out_rgb, current_time, image_rgb.frame)
                    image_semseg.save_to_disk(seg_file)
                    image_rgb.save_to_disk(rgb_file)
                    image_collected += postprocess(seg_file, rgb_file)


                # Draw the display.
                if show_recording == 1:
                    fps = round(1.0 / snapshot.timestamp.delta_seconds)
                    draw_image(display, image_rgb)
                    draw_image(display, image_semseg, blend=True)
                    display.blit(
                        font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                        (8, 10))
                    display.blit(
                        font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                        (8, 28))
                    pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')