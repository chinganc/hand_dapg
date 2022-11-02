################ Uncommment this part to install mujoco210
sudo apt-get install -y libglew-dev patchelf libosmesa6-dev
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
################

pip install -r requirements.txt

git clone https://github.com/chinganc/mjrl.git
cd mjrl
pip install -e .
cd ..


git clone --branch v0.0.5 --recursive https://github.com/vikashplus/mj_envs.git
cd mj_envs  # pip install -e .
echo Try to install $PWD
export PYTHONPATH=$PYTHONPATH:$PWD
cd ..

pip install torch==1.7.1
pip install --upgrade mujoco_py