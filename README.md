# 图像场景图生成使用说明

## 1. 算法描述
该算法用于图像场景图生成(SGG)。该算法基于PyTorch框架开发，输入一张图片，算法会检测图片中人和物的交互关系，输出图像场景图。

## 2. 依赖及安装

CUDA版本: 11.3
其他依赖库的安装命令如下：

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

可使用如下命令下载安装算法包：
```bash
pip install -U mmkg-sgg
```

## 3. 使用示例及运行参数说明

**需要准备 pretrained_ckpt, labelmap_file, freq_prior 三个文件，可以通过 [mmkg-sgg](https://github.com/xu7yue/mmkg-sgg) 上所提供的链接进行下载**
* [google drive](https://drive.google.com/file/d/1_GE1K_hJ9-FU8p8MWfcDYbgFT3MCabwN/view?usp=share_link)

```python
from mmkg_vrd import SGG

sgg = SGG(
    pretrained_ckpt='custom_io/ckpt/RX152FPN_reldn_oi_best.pth',
    labelmap_file='custom_io/ji_vrd_labelmap.json', 
    freq_prior='custom_io/vrd_frequency_prior_include_background.npy', 
)

sgg.det_and_vis(
    img_file='custom_io/imgs/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg',
    save_file='custom_io/out/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.reldn_relation.jpg')

print('all done')

```
