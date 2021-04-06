## Start Carla Unreal Engine Editor
Open a terminal window and do
```bash
$ cd ~/carla-0.9.6
$ make launch
```
Then clicks the **play** button to start the server.


## Launch Python Client and test connection

To see a specific town
```bash
$ cd ~/simulation4detection_v2/carla-0.9.6/PythonAPI
$ python3 examples/selectTown.py -town=1 # Load town01 in the simulator, you can choose from town [1, 2, 3, 4, 5]
```

To run some vehicles and save camera images
```bash
$ cd ~/simulation4detection_v2/carla-0.9.6/PythonAPI
$ python3 examples/saveImages.py # Create a autopilot vehicle and save rgb and segmentation camera images
```

To change the weather to be very cloudy
```bash
$ cd ~/simulation4detection_v2/carla-0.9.6/PythonAPI
$ python3 examples/changeWeather.py
```