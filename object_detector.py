def detect_objects(frame_path, model):
    # Perform inference
    results = model(frame_path)
    detections = results[0]  # Get the first result from the list

    # Extract bounding boxes, confidence scores, and class names
    boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = detections.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = detections.boxes.cls.cpu().numpy()  # Class IDs

    # Map class IDs to names
    class_names = [model.names[int(cls_id)] for cls_id in class_ids]

    # Create a structured output
    return {
        "boxes": boxes,
        "confidences": confidences,
        "class_names": class_names
    }
