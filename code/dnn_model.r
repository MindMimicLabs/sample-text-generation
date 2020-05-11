require(keras)
require(yaml)

create_dnn <- function(max_sentence_length, unique_vocab) {
    
  FLAGS <- yaml.load_file('./dnn_model.yml')
  
  model <-
    keras_model_sequential() %>% 
    layer_lstm(
      units = FLAGS$units,
      input_shape = c(max_sentence_length, unique_vocab),
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg, l2 = FLAGS$reg)) %>%
    layer_dense(unique_vocab) %>%
    layer_activation("softmax")
  
  compile(
    model,
    loss = "categorical_crossentropy", 
    optimizer = optimizer_nadam(lr = FLAGS$lr))
  
  model
}
