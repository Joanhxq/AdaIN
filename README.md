# pytorch-AdaIN

This is an unofficial pytorch implementation of[ Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf).

Original torch implementation from the author can be found [here](https://github.com/xunhuang1995/AdaIN-style).

## Requirements

- Python 3.5+
- Pytorch 0.4+
- TorchVision
- tqdm

Anaconda environment recommended here!

(optional)

- GPU environment for training

# Usage

### Train

`python train.py args`

Possible ARGS are:

- `-h, --help` Show this help message and exit;
- `--content_dir CONTENT_DIR` Directory path to amount of content images;
- `--style_dir STYLE_DIR` Directory path to amount of style images;
- `--save_models SAVE_MODELS` Path to save the trained model (default=`save_models`);
- `--batch_size BATCH_SIZE` The size of batch to train (default=`8`);
- `--alpha ALPHA` a smooth transition between content-similarity and style-similarity can be observed by changing α from 0 to 1.0 (default=`1.0`);
- `--lambda_weight LAMBDA_WEIGHT` The degree of style transfer can be controlled during training by adjusting the style weight λ (default=`10.0`);
- `--lr LR` The learning rate of Adam (default=`1e-4`);
- `--lr_decay LR_DECAY` The decay rate of learning rate (default=`5e-5`);
- `--max_epoch MAX_EPOCH` Number of iterations (default=`160000`);
- `--save_model_interval SAVE_MODEL_INVTERVAL` The interval epoch to save trained model (default=`10000`)

------

Use `--content_dir` and `--style_dir` to provide the respective directory to the content and style images.

```
python train.py --content_dir input/content --style_dir input/style
```

### Test

test.py

`python test.py args`

Possible ARGS are:

- `-h, --help` Show this help message and exit;
- `--content CONTENT` File path to content image;
- `--style STYLE` File path to style image;
- `--content_dir CONTENT_DIR` Directory path to a batch of content images;
- `--style_dir STYLE_DIR` Directory path to a batch of style images;
- `--img_size IMG_SIZE` Minimum size for images, keeping the original size if given 0 (default is `512`);
- `--output_dir OUTPUT_DIR` Directory to save the stylized images (default is `output`);
- `--style_interpolation_weights STYLE_INTERPOLATION_WEIGHTS` The weight for blending the multiple style images;
- `--decoder DECODER` Path for the arguments of decoder (default is `models/decoder.pth`);
- `--alpha ALPHA` a smooth transition between content-similarity and style-similarity can be observed by changing α from 0 to 1.0 (default is `1.0`);
- `--perserve_color PERSERVE_COLOR` If specified, preserve color of the content image (action=`store_true`);
- `--crop CROP` do center crop to create squared image (action=`store_true`);

---

Use `--content` and `--style` to provide the repective path to the content and style image.

```
python test.py --content input/content/brad_pitt.jpg --style input/style/sketch.png
```

You can also run the code on directories of content and style images using `--content_dir` and `--style_dir`. It will save every possible combination of content and styles to the output directory.

```
python test.py --content_dir input/content --style_dir input/style
```

This is an example of mixing four styles by specifying `--style` and `--style_interpolation_weights` option.

```
python test.py --content input/content/avril.jpg --style input/style/antimonocromatismo.jpg,input/style/asheville.jpg,input/style/sketch.png,input/style/impronte_d_artista.jpg --style_interpolation_weights 1,1,1,1 --crop
```

## Results

<figure class="half">

<img src="output\avril_interpolation.png" width="300px" height="300px">

<img src="output\brad_pitt_stylized_by_sketch.png" width="300px" height="300px">

<img src="output\avril_stylized_by_trial.png" width="484px" height="300px">

<img src="output\newyork_stylized_by_brushstrokes.png" width="484px" height="300px">

</figure>