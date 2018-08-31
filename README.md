## CS194-26 Image Manipulation and Computational Photography


### Project 1: Colorizing the Prokudin-Gorskii Photo Collection

This program will colorize digitized RGB glass plate negatives downloaded from
the [library of congress](https://www.loc.gov/collections/prokudin-gorskii/?sp=1).


### Run

To run you simply need to run the main.py file and pass it the path to the file.

```
python3 main.py --imname='images/input/emir.tif'
```

The output file will be stored in the same folder as the input file and will have the extension ```_out.jpg```

### Arguments

You can set the ```verbose``` argument to true and then the intermediate images will also be saved. These include a naive alignment (```_naive.jpg```),, an aligned without cropping (```_aligned.jpg```), an aligned with cropping but no contrasting (```_cut.jpg```),, and the final version with alignment, cropping, and contrasting (```_out.jpg```)

```
python3 main.py --imname='images/input/emir.tif' --verbose=True
```

You can also specify a window if you desire to restrict the search space over which to align. Here you need to pass a lower boundary ```x``` and an upper boundary ```y```.

```
python3 main.py --imname='images/input/emir.tif' --verbose=True --x=-120 --y=120
```


##### Enjoy :)

### Author

Morad Shefa
