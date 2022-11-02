pip install -r requirement.txt

git clone https://github.com/chinganc/mjrl.git
cd mjrl
pip install -e .
cd ..


git clone --branch v0.0.5 --recursive https://github.com/vikashplus/mj_envs.git
cd mj_envs
pip install -e .
cd ..

pip install torch==1.7.1
pip install --upgrade mujoco_py