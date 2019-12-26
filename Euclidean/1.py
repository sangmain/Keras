import cv2
import numpy as np

img = cv2.imread("./3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, bin_img = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("aaa", dst)
# cv2.waitKey(0)


# kernel = np.ones((3, 3), np.uint8)
# proc = cv2.erode(dst, kernel, iterations=1)
# proc = cv2.dilate(proc, kernel, iterations=1)  #// make dilation image

# # cv2.imshow("aaa", a)
# # cv2.waitKey(0)


# proc = cv2.dilate(proc, kernel, iterations=1)  #// make dilation image
# proc = cv2.erode(proc, kernel, iterations=1)

# proc = cv2.dilate(proc, kernel, iterations=1)  #// make dilation image
# bin_img = cv2.dilate(proc, kernel, iterations=1)  #// make dilation image

# cv2.imshow("aaa", bin_img)
# cv2.waitKey(0)

print("start")
### prefix sum
def prefix(array):
    num = 1
    result = []

    for i in range(array.shape[0]):
        if array[i] == 1:
            result.append(num)
            num += 1

        else:
            result.append(0)
            num = 1

    return result

### postfix sum
def postfix(array):
    array = array[::-1]
    num = 1
    result = []

    for i in range(array.shape[0]):
        if array[i] == 1:
            result.append(num)
            num += 1

        else:
            result.append(0)
            num = 1

    result = result[::-1]
    return result
 
bin_img = bin_img.reshape(-1, bin_img.shape[0])
print(bin_img.shape)
row_pre= np.zeros((bin_img.shape)); row_post = np.zeros((bin_img.shape)); col_pre = np.zeros((bin_img.shape)); col_post = np.zeros((bin_img.shape));

### 열
for i in range(bin_img.shape[0]):
    row = bin_img[i, :]
    row_pre[i, :] = prefix(row)
    row_post[i, :] = postfix(row)
    # print(bin_img[i, :])
    print(prefix(row))
    print()

# for i in range(bin_img.shape[0]):
#     for j in range(bin_img.shape[1]):
#         if row_post[i, j] != 0.0:
#             print(row_post[i,j])


### 비교
row_result = np.zeros((bin_img.shape))
for i in range(bin_img.shape[0]):
    for j in range(bin_img.shape[1]):
        if bin_img[i, j] == 0:
            continue
        row_result[i,j] = min(row_pre[i,j], row_post[i,j])

### 행
for i in range(bin_img.shape[1]):
    col = bin_img[:, i]
    col_pre[:,i] = prefix(col)
    col_post[:, i] = postfix(col)

### 비교
col_result = np.zeros((bin_img.shape))
for i in range(bin_img.shape[0]):
    for j in range(bin_img.shape[1]):
        if bin_img[i, j] == 0:
            continue
        col_result[i,j] = min(col_pre[i,j], col_post[i,j])

### 비교

result = np.zeros((bin_img.shape))
for i in range(bin_img.shape[0]):
    for j in range(bin_img.shape[1]):
        if bin_img[i, j] == 0:
            continue
        result[i,j] = min(row_result[i,j], col_result[i,j])

print(result)

# for i in range(bin_img.shape[0]):
#     for j in range(bin_img.shape[1]):
#         if bin_img[i, j] != 0:
#             print(result[i,j])

# cv2.imshow("result", result)
# cv2.waitKey(0)




