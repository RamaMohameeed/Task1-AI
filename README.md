# Bird vs. Drone Image Classification System  
*A Machine Learning Project using Teachable Machine, TensorFlow and Jira/Xray*

---

##  Overview

This repository showcases a deployable image classification system trained to distinguish between **Bird** and **Drone** images. The model is developed using **Google Teachable Machine**, exported in TensorFlow Keras format, and integrated with a Python-based inference pipeline using **OpenCV** and **NumPy**.  
Test management is professionally handled via **Jira + Xray**, simulating a real-world software QA workflow.

---

##  Demonstration

|  Input Image                                                                              |                                                       Output Image                          |
|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------
| ![test](https://github.com/user-attachments/assets/35dff5d1-9590-4099-98ac-ce5383a73e63) | ![output](https://github.com/user-attachments/assets/d75bd331-6d68-48a4-af89-d1a60aaf88b7)|


---

##  Key Features

-  **Binary Classification**: Bird or Drone
-  **No-Code Training**: Google Teachable Machine
-  **Deployment-Ready**: Uses Keras `.h5` model
-  **Visual Feedback**: Class label overlay on output image
-  **Console Logging**: Predicted class shown in terminal
-  **Full SDLC & QA**: Tested and documented using Xray and Jira

---

# Testing & Validation
Functional testing was executed using Xray Test Management in Jira.
The system passed all defined test cases under manual execution.

 Test Summary:
Test Case ID	Description	Status
TC-001	Classification of Drone Image	✅ Passed
TC-002	Accuracy and Output Verification	✅ Passed

📌 **Detailed execution logs, evidence screenshots, and test plans are available in the docs/ folder.**

