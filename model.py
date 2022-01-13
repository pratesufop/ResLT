import tensorflow

def shared_representation(num_feats = [32, 64], input_shape = (28,28,1)):
    
    """ Returns a tf.keras.Model that computes a feature representation for an input image
        Inputs:
            num_feats: a list containing the number of output dims (feat. maps) for the convolutational layers sequentially
            input_shape : shape of the input data, as we are using the MNIST images (28,28,1)
        Returns:
            tf.keras.Model that is later used to compute the shared feature representation for an input image
    """
    
    inputs = tensorflow.keras.Input(shape=input_shape, name = 'input')
    
    x = inputs
    
    for num in num_feats:
        x = tensorflow.keras.layers.Conv2D(num, (3,3), padding = "same")(x)
        x = tensorflow.keras.layers.LeakyReLU(0.2)(x)
        x = tensorflow.keras.layers.MaxPool2D((2,2))(x)
    
    feats = x
    
    return tensorflow.keras.Model(inputs=inputs, outputs=feats, name = 'shared_representation')
    

def specialized_representation(input_shape, num_outputs, nm):
    """ Returns a tf.keras.Model that computes a feature representation for an input feature map
        Inputs:
            num_outputs: the number of classes in the dense layer
            input_shape : shape of the input data, in this case, a feature map (batch_size, h, w, C)
            nm: name of the layer
        Returns:
            tf.keras.Model that is later used to compute the output before softmax activation
    """
    
    inputs = tensorflow.keras.Input(shape=input_shape[1:], name = 'input')
    
    feat = tensorflow.keras.layers.Conv2D(32, (1,1), padding = "same")(inputs)
    feat = tensorflow.keras.layers.GlobalAveragePooling2D()(feat)
    out = tensorflow.keras.layers.Dense(num_outputs, activation = 'linear')(feat)
    
    return tensorflow.keras.Model(inputs=inputs, outputs=out, name = nm)
    
def ResLT_model(num_outputs = 10, input_shape = (28,28,1), num_feats = [32, 64]):
    """ Returns two tf.keras.Models one to train the ResLT model and one to predict the final result using the fusion path
        Inputs:
            num_outputs: the number of classes in the dense layer
            input_shape: shape of the input data, as we are using the MNIST images (28,28,1)
            num_feats: a list containing the number of output dims (feat. maps) for the convolutational layers sequentially
        Returns:
            model: tf.keras.Model that is later used to train the ResLT model
            test_model : tf.keras.Model that is later used to predict the output using the fusion path
    """
    input_hmt = tensorflow.keras.Input(shape= input_shape, name = 'input_hmt')
    input_mt = tensorflow.keras.Input(shape= input_shape, name = 'input_mt')
    input_t = tensorflow.keras.Input(shape=input_shape, name = 'input_t')
    
    # shared feature
    feat_model = shared_representation(num_feats, input_shape)
    
    # speciliazed feature
    feat_hmt = feat_model(input_hmt)
    sp_hmt = specialized_representation(feat_hmt.shape, num_outputs, nm = 'hmt_model')
    out_hmt = sp_hmt(feat_hmt)
    
    feat_mt = feat_model(input_mt)
    sp_mt = specialized_representation(feat_mt.shape, num_outputs, nm = 'mt_model')
    out_mt = sp_mt(feat_mt)
    
    feat_t = feat_model(input_t)
    sp_t = specialized_representation(feat_t.shape, num_outputs, nm = 't_model')
    out_t = sp_t(feat_t)
    
    out_fusion = tensorflow.keras.layers.Activation('softmax', name = 'fusion')(out_hmt + sp_mt(feat_hmt) + sp_t(feat_hmt))
    
    out_hmt = tensorflow.keras.layers.Activation('softmax', name = 'hmt')(out_hmt)
    out_mt = tensorflow.keras.layers.Activation('softmax', name = 'mt')(out_mt)
    out_t = tensorflow.keras.layers.Activation('softmax', name = 't')(out_t)
    
    model = tensorflow.keras.Model(inputs=[input_hmt, input_mt, input_t], outputs=[out_fusion, out_hmt,  out_mt, out_t], name = 'mnist_model')
    # prediction model
    test_model = tensorflow.keras.Model(inputs=input_hmt, outputs= out_fusion, name = 'mnist_prediction')
    
    return model, test_model
