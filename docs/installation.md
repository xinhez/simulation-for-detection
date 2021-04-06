## 1. Install required tools
Install pre-requisite tools listed in [carla-tools.sh](bin/carla-tools.sh) by running
```bash
$ cd
$ git clone https://gitlab.com/cmertz/simulation4detection_v2.git
$ cd simulation4detection_v2
$ ./bin/carla-tools.sh
```
Then follow the instructions as they show up in your terminal.

*Run `chmod +x bin/carla-tools.sh` before `./bin/carla-tools.sh` if having an error saying the file is not executable.

## 2. Install Unreal Engine 4.24
Get access to the private repository by following [this link](https://www.unrealengine.com/en-US/ue4-on-github).
```bash
$ git clone --depth=1 -b 4.24 https://github.com/EpicGames/UnrealEngine.git ~/UnrealEngine_4.24
$ cd ~/UnrealEngine_4.24
$ ./Setup.sh && ./GenerateProjectFiles.sh && make # takes ~45 minutes
```
  *Time estimates are based on a machine running i7-7700 and GTX 1080 Ti.*

## 3. Install Carla
```bash
$ git clone --branch 0.9.9 https://github.com/carla-simulator/carla ~/carla-0.9.9
$ cd ~/carla-0.9.9
$ ./Update.sh # takes ~45 minutes
$ export UE4_ROOT=~/UnrealEngine_4.24
$ make launch # takes ~30 minutes if running for the first time 
```
The UnrealEngine editor should automatically open.

  *Time estimates are based on a machine running i7-7700 and GTX 1080 Ti.*

## 4.  Prepare PythonClient

##### Install Carla Python requirements
Carla provides us with Python support. You can find a `PythonClient` folder in `~/carla`. 

Cross-version `PythonClient` and Carla won't work together.

```bash
$ cd ~/carla
$ make PythonAPI.3 # This will generate an egg file at ~/carla-0.9.9/PythonAPI/carla/dist/carla-0.9.9-py3.6-linux-x86_64.egg
$ cp PythonAPI/carla/dist/carla-0.9.9-py3.6-linux-x86_64.egg ~/simulation4detection_v2/carla/PythonAPI/carla/dist/carla-0.9.9-py3-linux-x86_64.egg # create a copy of this egg file in the simulation4detection_v2 repo
# You might have to change the generated filename carla-0.9.9-py3.6-linux-x86_64.egg in the command above
$ pip3 install --user pygame numpy networkx
```

## 5. Troubleshooting
In the case of "Make sure you build ShaderCompileWorker", install ShaderCompileWorker manually by
```bash
$ ~/UnrealEngine_4.24/Engine/Build/BatchFiles/Linux/Build.sh ShaderCompileWorker Linux Development # -verbose
```

## (Caution) Update Carla
Download the latest version of carla. Note in the newer versions Carla could potentially be updating their Python libraries, which means some scripts in this project could fail to run.
```bash
$ cd ~/carla
$ git pull
$ ./Update.sh
$ make launch
```