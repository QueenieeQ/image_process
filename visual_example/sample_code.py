#!/usr/bin/python3
# Straightforward instructional implementation of Hough transform: how to find lines in an image?
# The image is tranformed to Hough space.
# The maxima in Hough space coincide with angles and perpendicular distances of straight lines in 
# the input image origin (top left here).
# See: https://en.wikipedia.org/wiki/Hough_transform
# The smart and fast way to do this would be with opencv:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# larsupilami73

import os
from scipy import *
import matplotlib.pyplot as plt

N = 100	#100x100 image size


#make straight line in an array
def StraightLine(X,m,c):
	if abs(m)<=1:
		for n in range(X.shape[0]):
			k = round(n*m + c)
			if (k<N) and (k>=0): A[int(k),n] = 1.0
	else: # to avoid lines with gaps
		for n in range(X.shape[0]):
			k = round((n-c)/m)
			if (k<N) and (k>=0): A[n,int(k)] = 1.0

#the Hough transform
#for simplicity, keep the Hough space same dimensions as the input image
#assuming square image
def HoughTransform(X, threshold=0.5):
	D = X.shape[0] 
	H = zeros((D,D))
	points=where(X>threshold)
	pxs = points[1] # watch out!
	pys = points[0]
	for px,py in zip(pxs,pys):
		for d in range(D):
			tetha = pi*d/D
			rho = round(px*cos(tetha) + py*sin(tetha)) #see: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html		
			if rho<D and rho>=0:
				H[int(rho),d] +=1.0	#row=distance,column=angle
			#print(px,py,d,rho)
	
	H /=H.max() #normalize [0..1]
	return H
	

#temp directory for the frames
if not os.path.exists('./frames'): 
	os.mkdir('./frames')
try:
	os.system('rm ./frames/frame*.png')
except:
	pass

#make straight lines, y=mx+c
MC = []
for c in range(10,90,5): MC.append((0,c))
for c in range(90,50,-5): MC.append((0,c))
for m in linspace(0,5,25): MC.append((m,50))
for m in linspace(5,-2,50): MC.append((m,50))
for c in range(50,200,5): MC.append((-2,c))
for k in range(10): MC.append((-2,200))

fig, (ax1,ax2) = plt.subplots(1,2, sharey=False,figsize=(6.4,3.2), dpi=150)
plt.subplots_adjust(wspace=0.4)
for k,(m,c) in enumerate(MC):
	#make the input image
	A =  zeros((N,N)) 
	StraightLine(A,m,c)
	#do the transform
	HA = HoughTransform(A) 

	#make the input image
	ax1.imshow(A,cmap=plt.cm.gray)
	ax1.set_title(r'Input image: $y=mx+c$')
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_xticks([0,25,50,75,100])
	ax1.set_yticks([0,25,50,75,100])
	ax1.text(1,97,r'$m=%.1f$   $c=%.1f$' %(m,c), color='lightgreen')

	#make the Hough space image
	ax2.imshow(pow(HA,0.65),cmap=plt.cm.afmhot) # x^(1/a) >= x in ]0,1] for a>1, so makes pixels clearer in Hough space
	ax2.set_title(r'Hough space: $(d,\theta)$')
	ax2.set_ylabel('Perp. dist. to origin [pixels]')
	ax2.set_xlabel('Angle [degrees]')
	ax2.set_xlim(0,N)
	
	#show where the maximum is (for only one line)
	#for more lines, we would select several local maxima in Hough space
	ym,xm = where(HA==HA.max())
	ym = ym[0]
	xm= xm[0]
	
	ax2.scatter([xm],[ym],s=100,facecolors='none',marker='o',alpha=0.6,linewidths=1.5,edgecolors='lightgreen')
	
	#add angle/distance text
	ax2.text(1,97,r'$d=%.1f$   $\theta=%.1f ^{\circ}$' %(ym,180*xm/N), color='lightgreen')
	
	xt = [x for x in linspace(0.,N,5)]
	ax2.set_xticks(xt)
	ax2.set_xticklabels(['%.1f'%(180*x/N) for x in xt]) #position to angle
	ax2.set_yticks([0,25,50,75,100])

	#plt.tight_layout()
	plt.savefig('./frames/frame%05d.png' %k)
	ax1.clear()
	ax2.clear()
	print('Frame number: %d/%d' %(k+1,len(MC)))


#now assemble all the pictures in a movie
#print('converting to gif!')
#os.system('convert -delay 40 -loop 0 ./frames/*.png fb_tot.gif')
#or to mp4
print('converting to mp4!')
os.system("ffmpeg -y -r 5 -i ./frames/frame%05d.png -c:v libx264 -vf fps=10 hough.mp4")