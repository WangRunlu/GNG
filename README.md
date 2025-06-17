
# Roadmap-Learning-with-Topology-Informed-Growing-Neural-Gas
**Contact: saroyam@oregonstate.edu**

## About
This repository generates navigation roadmaps from probabilistic occupancy maps of uncertain and cluttered environments. 

## Paper
```
@article{saroya2021roadmap,
  title={Roadmap Learning for Probabilistic Occupancy Maps with Topology-Informed Growing Neural Gas},
  author={Saroya, Manish and Best, Graeme and Hollinger, Geoffrey A},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={3},
  pages={4805--4812},
  year={2021},
  publisher={IEEE}
}
```
**Abstract**
We address the problem of generating navigation roadmaps for uncertain and cluttered environments represented with probabilistic occupancy maps. A key challenge is to generate roadmaps that provide connectivity through tight passages and paths around uncertain obstacles. We propose the topology-informed growing neural gas algorithm that leverages estimates of probabilistic topological structures computed using persistent homology theory. These topological structure estimates inform the random sampling distribution to focus the roadmap learning on challenging regions of the environment that have not yet been learned correctly. We present experiments for three real-world indoor point-cloud datasets represented as Hilbert maps. Our method outperforms baseline methods in terms of graph connectivity, path solution quality, and search efficiency. Compared to a much denser PRM*, our method achieves similar performance while enabling a 27× faster query time for shortest-path searches.

## Requirements
- [Gudhi](https://gudhi.inria.fr/python/latest/installation.html)
- [Neupy](http://neupy.com/pages/installation.html)

## Usage
```
需要的，你必须进入你自己单独创建的docker环境，以避免干扰服务器上的其他用户。

正确的方式：

1. 进入服务器：

ssh bdi@101.200.33.217 -p 52103

2. 确保数据集已下载到对应目录（服务器中）：

cd ~/GNG/dataset
wget https://raw.githubusercontent.com/manishsaroya/GNG/master/dataset/mapdata_4798.pickle

3. 进入你自己的docker容器环境（不会干扰其他用户）：

cd ~/GNG
docker run --rm -it -v $PWD:/workspace gng:py36-full bash

这一步会进入你自己的docker容器内的独立环境，所有依赖和包的安装都会局限于这个容器中，不会污染服务器的环境。

4. 在docker容器中运行代码：

PYTHONPATH=. python persistence/gng_neupy_run.py --map_type 4798

退出容器后，这些变动也不会影响服务器或他人的环境。

因此一定要使用你自己的docker环境来运行，避免直接在服务器的全局环境里操作，以防干扰到其他人。
python persistence/gng_neupy_run.py
```

## Roadmap
   ![](https://github.com/manishsaroya/GNG/blob/master/gng.gif)
   conda activate gng_py36

PYTHONPATH=. python persistence/gng_neupy_run.py --map_type intel

PYTHONPATH=. python persistence/gng_neupy_run.py --map_type bhm
