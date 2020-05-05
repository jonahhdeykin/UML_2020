This code implements hierarchical LDA with a fixed depth tree and a
stick breaking prior on the depth weights.  An infinite-depth tree can
be approximated by setting the depth to be very high.  This code
requires that you have installed the GSL package.

The input format of the data is the same as in the [LDA-C package](https://github.com/Blei-Lab/lda-c).
Each line contains

 [# of unique terms] [term #] : [count] ...

The settings file controls various parameters of the model.  There are
several settings files contained in this directory.


IMPORTANT:

I hope that this code is useful to you, but please note that this code
is UNSUPPORTED.  Do not email me (David Blei) with questions.  I like posting as
much code as possible, but I unfortunately do not have the time to
support all of it.  (This paragraph is my solution to the problem.)

HLDA-C is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version.

LDA-C is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.




