To make any of the three examples in this 
folder, compile and link each with webgui.c
from the parent directory. For example, to make
the main example named example.c:

gcc example.c ../webgui.c -o example

Afterward, run example from the command line 
and then open a web browser and enter the URL

http://localhost:15000

=======================

example.c demonstrates command buttons and WebGL 3D
keyboard.c demonstrates animation, keyboard capture, and pixmap 2D
mouse.c demonstrates mouse capture and WebGL 3D