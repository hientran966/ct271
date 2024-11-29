import argparse
import cv2
import time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path input image on which we'll perform inpainting")
ap.add_argument("-m", "--mask", type=str, required=True,
	help="path input mask which corresponds to damaged areas")
ap.add_argument("-a", "--method", type=str, default="telea",
	choices=["telea", "ns"],
	help="inpainting algorithm to use")
ap.add_argument("-r", "--radius", type=int, default=3,
	help="inpainting radius")
args = vars(ap.parse_args())

flags = cv2.INPAINT_TELEA
if args["method"] == "ns":
	flags = cv2.INPAINT_NS

image = cv2.imread(args["image"])
mask = cv2.imread(args["mask"])
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

num_runs = 10
times = []
output = None

for i in range(num_runs):
    start_time = time.time()
    output = cv2.inpaint(image, mask, args["radius"], flags=flags)
    end_time = time.time()
    times.append(end_time - start_time)

average_time = np.mean(times)
print(f"Average inpainting time over {num_runs} runs: {average_time:.4f} seconds.")

text = f"Avg Time: {average_time:.4f} sec"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)
thickness = 2
position = (300, 100)
cv2.putText(output, text, position, font, font_scale, color, thickness)

cv2.imshow("Image", image)
cv2.imshow("Output", output)
cv2.imwrite('result.png', output)
cv2.waitKey(0)