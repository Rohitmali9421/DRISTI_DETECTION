from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import logging
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
app.logger.info("Model loaded successfully!")

@app.route("/detect", methods=["POST"])
def detect_objects():
    if "file" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["file"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    try:
        results = model(image)
        descriptions = []
        detected_objects = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                object_name = model.names[class_id]

                # Bounding box
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                obj_width = x_max - x_min
                obj_height = y_max - y_min
                center_x = (x_min + x_max) / 2

                # ---- POSITION ESTIMATION ----
                relative_x = center_x / width
                if relative_x < 0.25:
                    position = "to your left"
                elif relative_x > 0.75:
                    position = "to your right"
                else:
                    position = "straight ahead"

                # ---- DISTANCE ESTIMATION ----
                height_ratio = obj_height / height  # better than area
                if height_ratio > 0.6:
                    distance = "very close"
                elif height_ratio > 0.3:
                    distance = "nearby"
                elif height_ratio > 0.15:
                    distance = "a bit ahead"
                else:
                    distance = "far away"

                # Natural description
                description = f"{object_name} {position}, {distance}"
                descriptions.append((object_name, description))
                detected_objects.append(object_name)

        # ---- PRIORITIZE HUMANS ----
        human_desc = [d for obj, d in descriptions if obj == "person"]
        other_desc = [d for obj, d in descriptions if obj != "person"]

        ordered_desc = human_desc + other_desc

        # ---- COMBINE INTO A MESSAGE ----
        if ordered_desc:
            if len(ordered_desc) == 1:
                message = f"I see a {ordered_desc[0]}."
            else:
                message = "I can see " + ", ".join(ordered_desc[:-1]) + f", and {ordered_desc[-1]}."
        else:
            message = "I don’t see anything around you."

        return jsonify({
            "objects": list(set(detected_objects)),
            "descriptions": [d for _, d in descriptions],
            "message": message
        })

    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": "Error processing image"}), 500


if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
