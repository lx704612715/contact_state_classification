# contact_state_classification
This is the repo for the robotics project "contact state classification using machine learning".

### Requirement
- numpy
- pandas
- tslearn (for Shapelets classifier)
- visdom
- facets
- tensorflow

### How to use
1. Download the dataset from https://drive.google.com/drive/folders/1GyiogHXgIxUiuVfkc2BFoljfxg-VJTo3 and put it in `contact_state_classification/tests/1908/hfv/csd_result/`
2. Check `contact_state_classification/config.py`, Add the features you want to use to `SIMPLE_FEATURES` and `COMPLEX_FEATURES` according to the table blow, `SIMPLE_FEATURES` being features with only one dimension and `COMPLEX_FEATURES` being features with more than one dimension.
3. Then check the other parameters. If `CIRCULAR_SPLICING` is on, the results of the local exploration are repeated once, to achieve a circular filling effect. `INTERPOLATION_METHOD` is used to control the method used for interpolation. `UPSAMPLING_RATE` is used to control the density of interpolation. `N_SPLITS` is used to control the size of the Stratified KFold partition.
4. Run `main.py`, and check result in the console or write to logfile using `cs_classifier::log_to_csv`
5. Visualize the result using `plotter.py`, you need to turn on the visdom server before running this script.


#### Resources
Google Drive: https://drive.google.com/drive/folders/1GyiogHXgIxUiuVfkc2BFoljfxg-VJTo3
Datasets e.g., RoboticsProject2510.pkl and demo video e.g., wiggling_example.mp4 are saved here.
#### Feature Value Explanation 
| Feature     | Description |
| :---        |    :----:   |
| label      | label for contact state       |
| lock_type   | different type of latch lock        |
| init_q   | initial joint state        |
| init_base_ht_ee   | initial end-effector(ee) pose in the Cartesian space on the robot base frame        |
| init_image   | None |
| init_rs_image   | initial image on the robot end-effector from a realsense(rs) camera      |
| init_usb_image   | initial image from an external perspective from a usb camera       |
| desired control values             |
| d_ee_velocity   | desired ee velocity on the ee frame      |
| d_ee_theta   | desired ee velocity but in the spherical coordinate        |
| d_ee_phi   | desired ee velocity but in the spherical coordinate        |
| observed state values             |
| dist   | observed moved distance after execution an exploration action (currently 12 actions for one exploration behavior in total)       |
| base_s_ee   | observed moved distance as form of directed vector after execution an exploration action        |
| error_q   | joint value difference. Once the robot finished a exploration action, it should move back to the initial joint pose (init_q). This feature shows how good the robot moved back to the initial pose       |
| obs_ee_theta   | observed ee velocity in the spherical coordinate after sending the d_ee_velocity to the robot        |
| obs_ee_phi   | observed ee velocity in the spherical coordinate after sending the d_ee_velocity to the robot        |
| init_ee_ft_robot   | observed force/torque sensor reading on the ee frame <strong> before </strong> controlling one exploration action         |
| curt_ee_ft_robot   | observed force/torque sensor reading on the ee frame <strong> after </strong> controlling one exploration action         |

### Background Knowledge
* Panda Robot Frame Information: 
  https://frankaemika.github.io/docs/control_parameters.html
  
  Frame 0 = Base Frame
  
  Frame F = End-Effector Frame
  
* Spherical Coordinate:
https://upload.wikimedia.org/wikipedia/commons/d/dc/3D_Spherical_2.svg
  
  We transform the translation (x, y, z) into spherical coordinate to reduce the dimensionality from 3(x, y, z) to 2 (theta, phi). As we are only interested in moving direction namely theta and phi
