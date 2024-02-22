clear;
close all
img=imread('(2).png');
img1=imresize(img,128/512);
imwrite(img1,'(1).png');

