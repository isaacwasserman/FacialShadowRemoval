# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import os
import urllib.request as urlreq
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.color import label2rgb
import json
import sklearn.mixture
import pickle as pkl
import torch

# %%
validation_paths = [
    "images/9156-004.png",
    "images/9162-031.png",
    "images/9165-009.png",
    "images/9169-020.png"
]
images = [(plt.imread(path) * 255).astype(np.uint8) for path in validation_paths]

uv_path = "uv_map.json" #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
uv_map_dict = json.load(open(uv_path))
uv_map = np.array([ (uv_map_dict["u"][str(i)],uv_map_dict["v"][str(i)]) for i in range(468)])

def update_readme():
    with open("README.md", "r") as f:
        lines = f.readlines()
    classical_header_index = lines.index("## Classical Approach\n")
    deeplearning_header_index = lines.index("## Deep Learning Approach\n")

    classical_figure_lines = []
    # add a line for each figure in /figures in alphabetical order
    for figure_name in sorted(os.listdir("figures")):
        classical_figure_lines.append(f"![](figures/{figure_name})\n")

    # replace the lines between the classical and deep learning headers with the new lines
    lines[classical_header_index+1:deeplearning_header_index] = classical_figure_lines
    # write the new lines to the README.md file
    with open("README.md", "w") as f:
        f.writelines(lines)


def display_images(images, save_name=None):
    pass
    # # display images as a row of subfigures
    # plt.figure(figsize=(2.5 * len(images), 2.5))
    # for i in range(len(images)):
    #     plt.subplot(1, len(images), i+1)
    #     plt.imshow(images[i])
    #     plt.axis('off')
    # plt.tight_layout()
    # if save_name is not None:
    #     plt.savefig(os.path.join("figures", save_name + ".png"))

def detect_face(image):
    # save face detection algorithm's url in haarcascade_url variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"
    # chech if file is in working directory
    if not (haarcascade in os.listdir(os.curdir)):
        urlreq.urlretrieve(haarcascade_url, haarcascade)
    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if len(faces) > 0:
        return faces[0]
    else:
        return np.array([0, image.shape[1], 0, image.shape[0]])

def display_bounding_boxes(images, bounding_boxes):
    image_templates = [image.copy() for image in images]
    for i, bounding_box in enumerate(bounding_boxes):
        (x,y,w,d) = bounding_box
        cv2.rectangle(image_templates[i],(x,y),(x+w, y+d),(255, 255, 255), 2)
    display_images(image_templates)

def detect_landmarks(image, bounding_box):
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "LFBmodel.yaml"
    # check if file is in working directory
    if not (LBFmodel in os.listdir(os.curdir)):
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    # create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.array([bounding_box]))
    if len(landmarks) > 0:
        return landmarks[0]
    else:
        return np.zeros((68, 2))

def display_landmarks(images, landmarks):
    image_templates = [image.copy() for image in images]
    for i, landmark_set in enumerate(landmarks):
        for landmark in landmark_set[0]:
            x, y = landmark
            cv2.circle(image_templates[i], (int(x), int(y)), 1, (255, 255, 255), 2)
    display_images(image_templates)

def detect_landmarks_3D(image):
    NUM_FACE = 1
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    else:
        return None
    
def convert_landmarks_to_2D(landmarks_3D):
    return np.array([(point.x, point.y) for point in landmarks_3D.landmark[0:468]])
    
def display_landmarks_3D(images, landmarkss, save_name=None):
    img_templates = [image.copy() for image in images]
    for i, landmarks in enumerate(landmarkss):
        mpDraw = mp.solutions.drawing_utils
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        img = img_templates[i]
        mpFaceMesh = mp.solutions.face_mesh
        mpDraw.draw_landmarks(img, landmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
        for id,lm in enumerate(landmarks.landmark):
            ih, iw, ic = img.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
    display_images(img_templates, save_name=save_name)

def detect_edges(image):
    # blur the image
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    # apply canny edge detection algorithm on the image
    return cv2.Canny(blurred_image, 25, 100)

def unwrap_face(image, landmarks):
    H_new,W_new = image.shape[0],image.shape[1]
    keypoints_uv = np.array([(W_new*x, H_new*y) for x,y in uv_map])
    keypoints = np.array([(W_new*point.x,H_new*point.y) for point in landmarks.landmark[0:468]])

    tform = PiecewiseAffineTransform()
    tform.estimate(keypoints_uv, keypoints)
    texture = warp(image, tform, output_shape=(H_new,W_new))
    texture = (255*texture).astype(np.uint8)

    return texture

def discard_eyes_eyebrows_and_nostrils(image, points):
    with open("triangulation.pkl", "rb") as f:
        triangles = pkl.load(f)
    triangles_to_discard = np.array([835, 760, 761, 757, 528, 527, 386, 387, 386, 387, 505, 506,
                        507, 535, 675, 771, 764, 763, 411, 412, 285, 286, 407, 406, 550, 551,
                        552, 684, 776, 777, 291, 294, 272, 547, 548, 543, 402, 253, 375, 376,
                        374, 249, 248, 373, 588, 589, 486, 342, 485, 587, 717, 716, 718,
                        720, 721, 613, 387, 386, 719, 714, 846, 845, 606, 605, 311, 312, 223,
                        224, 119, 109, 221, 350, 491, 809, 808, 728, 804, 803, 819, 818, 724,
                        844, 852, 888, 890, 889, 891,802, 469, 470, 467, 118, 640, 641, 638,
                        639, 637, 810, 821, 820, 811, 870, 869, 868, 854, 897, 896, 894, 895,
                        864, 865, 863, 862, 817, 646, 896, 866, 867, 861, 860, 859, 856, 860,
                        855, 816, 857, 858, 815, 816, 730, 729, 812, 813, 814, 629, 630, 643,
                        642, 629, 30, 120, 159, 158, 468, 466, 471, 604, 800, 801, 850, 885, 882,
                        883, 851, 879, 878, 877, 881, 880, 715, 884, 886, 156, 157, 230, 229,
                        227, 226, 635, 353, 636, 722, 723, 731, 732, 631, 632, 733, 735, 876,
                        875, 874, 873, 871, 872, 822, 824, 853, 893, 892, 734, 807, 806, 805,
                        887, 602, 603, 464, 465, 313, 228, 815, 816, 860, 629, 896])
    # find pixels within the triangles
    mask = np.ones(image.shape[:2], dtype=bool)
    for triangle_index in triangles_to_discard:
        triangle = points[triangles[triangle_index]][:, ::-1] * np.array([image.shape[1], image.shape[0]])
        if triangle[:, 0].max() > 125 and triangle[:, 0].max() < 150:
            print(triangle_index)
        mask = mask ^ skimage.draw.polygon2mask(image.shape[:2], triangle)
    return image * mask[:, :, np.newaxis]
        
def identify_shadow_distribution(unwrapped_face):
    # Get non-black pixels.
    face_pixels = unwrapped_face.reshape(-1, 3)
    face_pixels = face_pixels[np.sum(face_pixels, axis=1) > 0]
    lab_face_pixels = skimage.color.rgb2lab(face_pixels.reshape(1, -1, 3)).reshape(-1, 3)
    # Fit a Gaussian mixture model.
    gmm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type="diag")
    gmm.fit(lab_face_pixels)
    return gmm

def generate_shadow_probability(unwrapped_face, model):
    face_pixels = unwrapped_face.reshape(-1, 3)
    face_pixels = face_pixels[np.sum(face_pixels, axis=1) > 0]
    lab_face_pixels = skimage.color.rgb2lab(face_pixels.reshape(1, -1, 3)).reshape(-1, 3)
    gmm = model
    probabilities = gmm.predict_proba(lab_face_pixels)[:, 0]
    predictions = probabilities > 0.5
    if np.mean(face_pixels[predictions == 0], axis=0)[0] < np.mean(face_pixels[predictions == 1], axis=0)[0]:
        probabilities = 1 - probabilities
    mask = np.zeros(unwrapped_face.shape[:2])
    mask[np.sum(unwrapped_face, axis=2) > 0] = probabilities
    mask[np.sum(unwrapped_face, axis=2) <= 0] = -1
    return mask

def generate_imputed_mean_color(unwrapped_face, shadow_probability):
    shadow_mask = (shadow_probability < 0.5).astype(np.uint8)
    shadow_mask_dilated = cv2.erode(shadow_mask, np.ones((11, 11), np.uint8), iterations=1)
    data = unwrapped_face * shadow_mask_dilated[:,:,np.newaxis]
    # create 256x256x2 array of xy coordinates
    xy = np.dstack(np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0])))
    # add as channels of data
    data = np.dstack([xy, data])
    data = data.reshape(-1, 5).astype(np.float32)
    mask = data[:, -3:].sum(axis=1) == 0
    poly = sklearn.preprocessing.PolynomialFeatures(1)
    X_train = poly.fit_transform(data[~mask, :-3])
    X_test = poly.transform(data[mask, :-3])
    data[mask, -3:] = sklearn.linear_model.LinearRegression().fit(X_train, data[~mask, -3:]).predict(X_test)

    imputed = data.reshape(256, 256, 5)[:,:,-3:]
    existing_pixels = unwrapped_face * shadow_mask[:,:,np.newaxis]
    add_mask = np.dstack([imputed.sum(axis=2) > existing_pixels.sum(axis=2)]*3)
    color = (imputed * add_mask + existing_pixels * (1 - add_mask)).clip(0,255).astype(np.uint8)
    color = cv2.medianBlur(color, 29)
    return color

def remove_shadow_naive(unwrapped_face, mean_color_map, shadow_probability, uniform=False):
    target_means = mean_color_map
    source_means = cv2.medianBlur(unwrapped_face, 29)
    shadow_mask = shadow_probability > 0.5
    if uniform:
        shadow_mask = np.ones(unwrapped_face.shape[:2]).astype(np.uint8)
    adjustment_factor = ((target_means).astype(np.float32) / (source_means).astype(np.float32))#.clip(0, 10)
    adjustment_factor[unwrapped_face.sum(axis=2) == 0] = 1
    mean_shadow_adjustment_factor = adjustment_factor[shadow_mask].mean()
    adjustment_factor = np.dstack([shadow_mask]*3) * (adjustment_factor - 1)
    adjustment_factor = cv2.GaussianBlur((adjustment_factor * 255).clip(0, 255).astype(np.uint8), (3, 3), 1) / 255
    unshadowed_face = (unwrapped_face * (1 + adjustment_factor)).clip(0, 255).astype(np.uint8)
    return unshadowed_face

def wrap_face(unwrapped_face, landmarks):
    H_new,W_new = unwrapped_face.shape[0],unwrapped_face.shape[1]
    keypoints_uv = np.array([(W_new*x, H_new*y) for x,y in uv_map])
    keypoints = np.array([(W_new*point.x,H_new*point.y) for point in landmarks.landmark[0:468]])

    tform = PiecewiseAffineTransform()
    tform.estimate(keypoints, keypoints_uv)
    wrapped = warp(unwrapped_face, tform, output_shape=(H_new,W_new))
    wrapped = (255*wrapped).astype(np.uint8)

    return wrapped

def blend_seamless(original_image, rewrapped_face, landmarks):
    noise = (np.random.normal(0, 1, rewrapped_face.shape) * 5).clip(0, 255).astype(np.uint8)
    noisy_rewrapped_face = (rewrapped_face.astype(int) + noise.astype(int)).clip(0, 255).astype(np.uint8)
    alpha = (rewrapped_face.sum(2) > 0).astype(np.uint8) * 255

    noisy_rewrapped_face = noisy_rewrapped_face.astype(int) + (original_image * ((255 - alpha[:,:,np.newaxis]) / 255))
    noisy_rewrapped_face = noisy_rewrapped_face.clip(0, 255).astype(np.uint8)

    alpha = cv2.erode(alpha, np.ones((5, 5), np.uint8), iterations=1)

    alpha = cv2.GaussianBlur(alpha, (51, 51), 0, borderType=cv2.BORDER_REFLECT)
    alpha = alpha[:,:,np.newaxis]

    src_luminance = noisy_rewrapped_face.sum(2)
    dst_luminance = original_image.sum(2)
    alpha[dst_luminance > src_luminance] = 255

    highest_landmark = 1
    lowest_landmark = 0
    leftmost_landmark = 1
    rightmost_landmark = 0
    for landmark in convert_landmarks_to_2D(landmarks):
        if landmark[1] < highest_landmark:
            highest_landmark = landmark[1]
        if landmark[1] > lowest_landmark:
            lowest_landmark = landmark[1]
        if landmark[0] < leftmost_landmark:
            leftmost_landmark = landmark[0]
        if landmark[0] > rightmost_landmark:
            rightmost_landmark = landmark[0]

    center = (np.mean([highest_landmark, lowest_landmark]) * original_image.shape[0], np.mean([leftmost_landmark, rightmost_landmark]) * original_image.shape[1])
    center = (int(center[1]) - 1, int(center[0]) - 1)
    seamless_blend = cv2.seamlessClone(noisy_rewrapped_face, original_image, alpha, center, cv2.NORMAL_CLONE)
    return seamless_blend

def get_eye_mouth(image, landmarks_2d):
    with open("triangulation.pkl", "rb") as f:
        triangles = pkl.load(f)
    triangles_to_include = np.array([835, 760, 761, 757, 528, 527, 386, 387, 386, 387, 505, 506,
                        507, 535, 675, 771, 764, 763, 411, 412, 285, 286, 407, 406, 550, 551,
                        552, 684, 776, 777, 291, 294, 272, 547, 548, 543, 402, 253, 375, 376,
                        374, 249, 248, 373, 588, 589, 486, 342, 485, 587, 717, 716, 718,
                        720, 721, 613, 387, 386, 719, 714, 846, 845, 606, 605, 311, 312, 223,
                        224, 119, 109, 221, 350, 491, 809, 808, 728, 804, 803, 819, 818, 724,
                        844, 852, 888, 890, 889, 891,802, 469, 470, 467, 118, 640, 641, 638,
                        639, 637, 810, 821, 820, 811, 870, 869, 868, 854, 897, 896, 894, 895,
                        864, 865, 863, 862, 817, 646, 896, 866, 867, 861, 860, 859, 856, 860,
                        855, 816, 857, 858, 815, 816, 730, 729, 812, 813, 814, 629, 630, 643,
                        642, 629, 30, 120, 159, 158, 468, 466, 471, 604, 800, 801, 850, 885, 882,
                        883, 851, 879, 878, 877, 881, 880, 715, 884, 886, 156, 157, 230, 229,
                        227, 226, 635, 353, 636, 722, 723, 731, 732, 631, 632, 733, 735, 876,
                        875, 874, 873, 871, 872, 822, 824, 853, 893, 892, 734, 807, 806, 805,
                        887, 602, 603, 464, 465, 313, 228, 815, 816, 860, 629, 896])
    # find pixels within the triangles
    mask = np.ones(image.shape[:2], dtype=bool)
    for triangle_index in triangles_to_include:
        triangle = landmarks_2d[triangles[triangle_index]][:, ::-1] * np.array([image.shape[1], image.shape[0]])
        if triangle[:, 0].max() > 125 and triangle[:, 0].max() < 150:
            print(triangle_index)
        mask = mask ^ skimage.draw.polygon2mask(image.shape[:2], triangle)
    return image * ~mask[:, :, np.newaxis]

def get_face_mask(face_image, landmarks):
    landmarks_2d = np.array([(256*point.x,256*point.y) for point in landmarks.landmark[0:468]])
    with open("triangulation.pkl", "rb") as f:
        triangles = pkl.load(f)
    mask = np.ones((256, 256), dtype=bool)
    for triangle in triangles:
        mask = mask ^ skimage.draw.polygon2mask((256, 256), landmarks_2d[triangle][:, ::-1])
    return ~mask[:,:,np.newaxis]

def inpaint_seam(composited_image, shadow_mask, seam=None):
    if seam is None:
        shadow_contour = cv2.findContours((shadow_mask > 0.99).astype(np.uint8) + (shadow_mask < 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        shadow_contour_map = np.zeros(composited_image.shape[:2])
        cv2.drawContours(shadow_contour_map, shadow_contour, -1, 1, 10)
        shadow_contour_map = (shadow_contour_map == 1).astype(np.uint8)
        seam = shadow_contour_map
    inpainted = cv2.inpaint(composited_image, seam, 20, cv2.INPAINT_TELEA)
    return inpainted

# %%
face_images = images
display_images(face_images, save_name="01_original")

face_landmarks = [detect_landmarks_3D(image) for image in face_images]
face_masks = [get_face_mask(face_image, landmarks) for face_image, landmarks in zip(face_images, face_landmarks)]
masked_face_images = [face_image * mask for face_image, mask in zip(face_images, face_masks)]
display_images(masked_face_images, save_name="02_isolated_faces")

shadow_masks = [generate_shadow_probability(masked_face_image, identify_shadow_distribution(masked_face_image)) > 0.5 for masked_face_image in masked_face_images]
display_images(shadow_masks, save_name="03_shadow_masks")

eroded_shadow_masks = shadow_masks
dilated_shadow_masks = [cv2.dilate(shadow_mask.astype(np.uint8), np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8), iterations=7) for shadow_mask in shadow_masks]
dilation_diffs = [(dilated_shadow_mask - eroded_shadow_mask) * face_mask[:,:,0] for dilated_shadow_mask, eroded_shadow_mask, face_mask in zip(dilated_shadow_masks, eroded_shadow_masks, face_masks)]
display_images(dilation_diffs, save_name="04_shadow_contours")

unshadowed_masked_face_images = [remove_shadow_naive(masked_face_image, generate_imputed_mean_color(masked_face_image, shadow_mask), shadow_mask, uniform=False) for masked_face_image, shadow_mask in zip(masked_face_images, shadow_masks)]
display_images(unshadowed_masked_face_images, save_name="05_unshadowed")

composited = [((1 - mask) * face_image + mask * unshadowed_masked_face_image).clip(0,255).astype(np.uint8) for mask, face_image, unshadowed_masked_face_image in zip(face_masks, face_images, unshadowed_masked_face_images)]
display_images(composited, save_name="06_composited")

inpaintings = [inpaint_seam(composite, shadow_mask, seam=dilation_diff) for composite, shadow_mask, dilation_diff in zip(composited, shadow_masks, dilation_diffs)]
display_images(inpaintings, save_name="07_inpainted")
display_images(face_images)

update_readme()

# %%
device = "cpu"
class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        Gx = torch.stack([Gx] * 3)
        Gy = torch.stack([Gy] * 3)
        G = torch.stack([Gx, Gy])
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)
        self.filter = self.filter.to(device)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class SobelLoss(torch.nn.Module):
    def __init__(self, image, prediction_mask):
        super(SobelLoss, self).__init__()
        self.base_image = torch.tensor(image, dtype=torch.float32).to(device)
        self.prediction_mask = torch.tensor(prediction_mask).to(device)
        self.sobel = Sobel().to(device)

    def forward(self, x):
        self.base_image[torch.where(self.prediction_mask)] = x.to(device)
        combined_gradient = self.sobel(self.base_image.permute(2, 0, 1)).squeeze(0)
        return torch.sum(torch.abs(combined_gradient))

def optimize_boundary(original_image, boundary_mask):
    image = original_image.copy()
    prediction_mask = boundary_mask.copy()
    parameter_pixels = torch.tensor(image[np.where(prediction_mask)], dtype=torch.float32).requires_grad_()
    optimizer = torch.optim.Adam([parameter_pixels], lr=0.2)
    num_iterations = 100
    sobel_loss = SobelLoss(image, prediction_mask).to(device)
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = sobel_loss(parameter_pixels)
        loss.backward(retain_graph=True)
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss.item()}")
    temp = torch.tensor(image.copy())
    temp[torch.where(torch.tensor(prediction_mask))] = parameter_pixels.clip(0,255).byte()
    image = temp.numpy()
    return image

# %%
optimize_boundary(composited[0], dilation_diffs[0])

# %%



