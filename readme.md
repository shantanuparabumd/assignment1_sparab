# CMSC848F - Assignment 1: Rendering Basics with PyTorch3D
## Authors

|Name|ID|Email|
|:---:|:---:|:---:|
|Shantanu Parab|119347539|sparab@umd.edu|

## Description

This package consist of the code and results for the assignment submission for CMSC848F.

**Open [writeup](/report/writeup.md.html) in a browser to access the writeup webpage.**

# 0. Setup 
- These instruction assume that you have completed all the installation and setup steps requrired.
- To recreate the results in this assignment, download the package and unzip into a folder.
- Move into the package `assignment1_sparab`.
- To render your first mesh run the following command, the results will be stored in the images folder.
- **Result:** `images/cow_render.jpg`
    
**Command:**
`python -m code.render_mesh`


---

# 1. Practicing with Cameras
- In this section we will try to create 2 effects 1. 360 Camera View and 2. Dolly Zoom


## 1.1 360 Renders
- This will generate multiple camera views and stich them together into a gif.
- This will give a visual effect of smooth 360 camera view
- **Result:** `images/render_360.gif`


**Command:**
`python -m code.render_360`

## 1.2 Dolly Zoom
- We will tr to create the dolly zoom effect
- **Result:** `images/dolly.gif`


**Command:**
`python -m code.dolly_zoom`


# 2. Practicing Meshes

## 2.1 Tetrahedron
- We will be creating a custom tetrahedron shape. The shape is stored in a form of vertices and faces in the `data/tetrahedron.obj` file
- We will be rendering this file and viewing it in 360.
- **Result:** `images/tetrahedron_render_360.gif`


**Command:**
`python -m code.render_360_tetrahedron`

## 2.2 Cube
- We will be creating a custom cube shape. The shape is stored in a form of vertices and faces in the `data/cube.obj` file
- We will be rendering this file and viewing it in 360.
- **Result:** `images/cube_render_360.gif`


**Command:**
`python -m code.render_360_cube`

# 3. Retexturing Meshes

- In this section we will be adding a color gradient to the cow rendering and viewing it in a 360 view.
- **Result:** `images/render_360_color.gif`


**Command:**
`python -m code.render_360_color`


# 4. Camera Transformation

- Here we will be doing camera transformations to get various views of the object.
- **Result:** `images/transform_cow_1.jpg`,`images/transform_cow_2.jpg`,`images/transform_cow_3.jpg`,`images/transform_cow_4.jpg`,`images/transform_cow_5.jpg`


**Command:**
`python -m code.camera_transforms`

# 5. Generic 3D Representation

## 5.1 Point Clouds from RGB-D (Plant)

- We will be using a point cloud representation with color values and viewing it in 3D.
- We will then combine multiple point clouds to get a dense 3D visulization in 360 view.
- **Result:** `images/render_360_plant_1.gif`,`images/render_360_plant_2.gif`,`images/render_360_plant_3.gif`


**Command:**
`python -m code.render_plant`

## 5.2 Parametric Representation (Torus)

- We will be using the parametric form of Torus to render a 3D point cloud using sampling and view it in 3D.
- **Result:** `images/torus_parametric_render_360.gif`


**Command:**
`python -m code.render_torus --render parametric`

## 5.3 Implicit Representation (Torus)

- We will be using the implicit form of Torus to render a 3D mesh and view it in 3D.
- **Result:** `images/torus_implicit_render_360.gif`


**Command:**
`python -m code.render_torus --render implicit`