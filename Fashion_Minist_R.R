install.packages("keras")
install.packages("tensorflow")
library(keras)
library(tensorflow)

# Load Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x / 255
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x / 255
test_labels <- fashion_mnist$test$y

# Reshape images for CNN
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Build CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

# Train the model
model %>% fit(train_images, train_labels, epochs = 10, validation_data = list(test_images, test_labels))

# Make predictions on two random test images
random_index1 <- sample(1:length(test_labels), 1)
random_index2 <- sample(1:length(test_labels), 1)

predictions <- model %>% predict_classes(test_images)

# Display images and predictions
par(mfrow = c(1,2))
image(test_images[random_index1,,], col = gray.colors(255), main = paste("Predicted:", predictions[random_index1]))
image(test_images[random_index2,,], col = gray.colors(255), main = paste("Predicted:", predictions[random_index2]))

