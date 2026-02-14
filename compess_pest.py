import tensorflow as tf

# Load original model
model = tf.keras.models.load_model("models/pest_model/pest_model.h5")

# Save again without optimizer
model.save("models/pest_model/pest_model.h5", include_optimizer=False)

print("Model compressed successfully!")