# Object Detection Based on Single Shot Multibox Detection Model
- Built SSD model, use VGG as a base network block and use several multiscale feature blocks to
generate anchor boxes and predict their categories and offsets, which achieve object detection.
- Defined loss and evaluation function, use L1 norm loss as anchor box offset loss and use the average
absolute error to evaluate the bounding box prediction results.