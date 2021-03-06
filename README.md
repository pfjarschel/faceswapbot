 # Faceswap Bot

This is a script that performs a face swap on two random images collected from SPB repository, built on top of the work done by Matthew Earl (@matthewearl).
After performing the face swap (both ways), it joins the resulting images and posts to Facebook, on the FaceswapBot 6656 page.

Feel free to use this code for fun or to create your own bot! I ~~intend to make~~ working on making it a little bit more user friendly ~~in the future~~, so it'll be easier to have fun with it!

## How to use

I suggest you use an advanced text editor, such as Atom, to view and edit the files as you need, create your own scripts, and then run it on a terminal emulator. You can, of course, use any IDE you want, or even windows notepad.

The file faceswaplib.py contains the main class used for face-swapping, all the relevant functions are there.
The faceswapbot.py contains the class that is responsible for fetching images and posting to Facebook. If you want just play around or make your own app with the faceswaplib, you don't need this, but it may still be useful!
Finally, faceswapbot_simple.py is the old script, mostly Matthew Earl's code, with some additions to post to FB. It should still work on its own.

### Prerequisites

For the script to work, you need:
* A valid installation of Python 3.X
* [dlib](https://pypi.org/project/dlib/) and its python bindings
* [The trained model](https://sourceforge.net/projects/dclib/postdownload)
* [OpenCV](https://opencv.org) and its python bindings
* [NumPy](http://www.numpy.org/)
* [Facebook python API](https://facebook-sdk.readthedocs.io/en/latest/install.html)



## Authors

* Matthew Earl (original code)
* Paulo Jarschel (modifications to get images from SPB and post to FB)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* @matthewearl for providing a great initial script and explanation
* Zbigniew Żołnierowicz for pointing me out to the aforementioned script
* [TheBot Appreciation Society](https://www.facebook.com/groups/botappreciationsociety/) for being such a great and inspiring group!
