import cv2
import numpy as np 
import dlib

#detector
detector = dlib.get_frontal_face_detector()
#shape predictor
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Image one
img1 = cv2.imread("messi.jpg")
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
mask1 = np.zeros_like(img_gray1)

faces1 = detector(img_gray1)
for face1 in faces1:
    landmarks1 = sp(img_gray1,face1)
    coor_landmarks1 = []
    for i in range(0,68):
        x= landmarks1.part(i).x
        y= landmarks1.part(i).y
        coor_landmarks1.append((x,y))
    # convert into numpy type 
    coor_points1 = np.array(coor_landmarks1, np.int32)
    convexhull1 = cv2.convexHull(coor_points1)

#delaunav triangulation 
rect = cv2.boundingRect(convexhull1)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(coor_landmarks1)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)


# Find the index of points of every vertex triangles
index_trangles = []
for t in triangles:
    vertex1 = (t[0], t[1])
    vertex2 = (t[2], t[3])
    vertex3 = (t[4], t[5])

    index_vertex_1 = np.where((coor_points1 == vertex1).all(axis = 1))[0][0]
    index_vertex_2 = np.where((coor_points1 == vertex2).all(axis = 1))[0][0]
    index_vertex_3 = np.where((coor_points1 == vertex3).all(axis = 1))[0][0]

    if index_vertex_1 is not None and index_vertex_2 is not None and index_vertex_3 is not None:
        index_trangles.append([index_vertex_1, index_vertex_2, index_vertex_3])


# Face 2
img2 = cv2.imread("person2.jpg")
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask2 = np.zeros_like(img_gray2)
faces2 = detector(img_gray2)
for face2 in faces2:
    landmarks2 = sp(img_gray2, face2)
    coor_landmarks2 = []
    for i in range(0,68):
        x = landmarks2.part(i).x
        y = landmarks2.part(i).y
        coor_landmarks2.append((x,y))
        
    # convert into numpy type 
    coor_points2 = np.array(coor_landmarks2, np.int32)
    convexhull2 = cv2.convexHull(coor_points2)


height, width, channels = img2.shape
img2_newface = np.zeros((height, width, channels), np.uint8)

# TODO: 
# 
lines_space_mask = np.zeros_like(img_gray1)
lines_space_new_face = np.zeros_like(img2)

# reuse the indexes triangles
for triangle in index_trangles:

    # no need to draw triangles in face 2
    # cv2.line(img2, pt1, pt2, (0, 0, 255), 1)
    # cv2.line(img2, pt3, pt2, (0, 0, 255), 1)
    # cv2.line(img2, pt1, pt3, (0, 0, 255), 1)
    
    # vertex triangle of the first face
    tr1_pt1 = coor_landmarks1[triangle[0]]
    tr1_pt2 = coor_landmarks1[triangle[1]]
    tr1_pt3 = coor_landmarks1[triangle[2]]

    # create a rectangle contain that triangle
    rect1 = cv2.boundingRect(np.array([tr1_pt1,tr1_pt2,tr1_pt3], np.int32))
    (x,y,w,h) = rect1
    croppedRect1 = img1[y:y+h, x:x+w]

    # the rectangle still contain the detail outside triangle, so we need to remove them
    tr1_mask = np.zeros((h,w), np.uint8)
    coor_3vertex_tri1_mask = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                        [tr1_pt3[0] - x, tr1_pt3[1] - y]],np.int32)
    # fill all pixel inside the triangle with white
    cv2.fillConvexPoly(tr1_mask,coor_3vertex_tri1_mask , 255)

    # cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    # cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    # cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    # croppedRect1 = cv2.bitwise_and(croppedRect1, croppedRect1, mask = lines_space_mask)

    # vertex triangle of the second face
    tr2_pt1 = coor_landmarks2[triangle[0]]
    tr2_pt2 = coor_landmarks2[triangle[1]]
    tr2_pt3 = coor_landmarks2[triangle[2]]

    # create a rectangle contain that triangle
    rect2 = cv2.boundingRect(np.array([tr2_pt1,tr2_pt2,tr2_pt3], np.int32))
    (x,y,w,h) = rect2
    croppedRect2 = img2[y:y+h, x:x+w]

    # the rectangle still contain the detail outside triangle, so we need to remove them
    tr2_mask = np.zeros((h,w), np.uint8)
    coor_3vertex_tri2_mask = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                        [tr2_pt3[0] - x, tr2_pt3[1] - y]],np.int32)
    # fill all pixel inside the triangle with white
    cv2.fillConvexPoly(tr2_mask,coor_3vertex_tri2_mask , 255)
    # croppedRect2 = cv2.bitwise_and(croppedRect2, croppedRect2, mask = tr2_mask)

    # warp 2 triangles
    coor_3vertex_tri1_mask = np.float32(coor_3vertex_tri1_mask)
    coor_3vertex_tri2_mask = np.float32(coor_3vertex_tri2_mask)
    M = cv2.getAffineTransform(coor_3vertex_tri1_mask, coor_3vertex_tri2_mask)
    warped_triangle = cv2.warpAffine(croppedRect1, M, (w,h))
    # TODO: wweird
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=tr2_mask)

    # Reconstrucing face2
    img2_newface_rect_area = img2_newface[y:y+h, x:x+w]
    # ngan khong cho vach trang xuat hien
    # TODO:
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_newface_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
   
    img2_newface_rect_area = cv2.add(img2_newface_rect_area, warped_triangle)
    img2_newface[y:y+h, x:x+w] = img2_newface_rect_area


# Face swapped

# Firstly we need to create a img2 but having noface
img2_face_mask = np.zeros_like(img_gray2)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)

# img2_nohead = cv2.fillConvexPoly(img2,convexhull2,0 )
# result = cv2.add(img2_nohead, img2_newface)

img2_face_mask = cv2.bitwise_not(img2_head_mask)


img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_newface)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)



# cv2.imshow("Image 1", img1)
# cv2.imshow("image2", img2)
cv2.imshow("s", seamlessclone)
cv2.waitKey(0)
cv2.destroyAllWindows()

