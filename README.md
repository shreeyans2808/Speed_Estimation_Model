# Speed Estimation using Birds Eye View of the road

In this Repository, I have made a speed estimation model using the help of **YOLO11l**, where I used `perspective_transform` on the road region to find out the distance each vehicle was moving by converting the coordinates from current pixels to distance travelled.

The logic behind this method is to find the matrix of constants required to transform the perspective to a **Birds Eye View** with the ratio of the width and length of the road and calculate the distance travelled by the vehicle in one second, to calculate the speed.

The `get_M` function in the code helps to get the transformation matrix, while the `get_perspective_transform` function helps to convert the coordinates of the given pixel to that required according to the distance travelled by mapping it to a second matrix of the given coordinates (width and height of the road).

I have also calculated the peak_traffic of the road during this process.

`Claude.ai` helped me with correcting some minor code errors which I was facing while trying to figure out the tracking process of each anchor (bottom_centre) of the vehicle.
