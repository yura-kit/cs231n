from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:], axis=0) #D = d_1 * ... * d_k
    out = x.reshape(N,D).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    pass
    N = x.shape[0]
    D = np.prod(x.shape[1:], axis=0)
    db = np.sum(dout, axis=0)
    dw = x.reshape(N, D).T.dot(dout)
    #print('X.shape=',x.shape)
    dx = dout.dot(w.T).reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    #x[x<0]=0
    #out = x
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    pass
    #x[x>0] = 1
    #x[x<=0] = 0
    dx = dout*(x>0)
    #dx = x*dout
    #print(dx)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        pass
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        cache = (x, x_norm, sample_mean, sample_var, gamma, eps)
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var


        '''
        #mini batch mean
        mu = 1/float(N)*np.sum(x, axis=0)
        x_mu = x - mu
        x_mu_quad = x_mu**2
        var = 1/float(N)*np.sum(x_mu_quad, axis=0)
        sqrt_var = np.sqrt(var + eps)
        inv_var = 1./sqrt_var
        norm = x_mu * inv_var
        norm2 = gamma * norm
        out = norm2 + beta

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_mean + (1 - momentum) * var
        cache = (mu, x_mu, x_mu_quad, var, sqrt_var, inv_var, norm, norm2,
                gamma, beta, x, bn_param)
        '''
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        #running_mean = bn_param['running_mean']
        #running_var = bn_param['running_var']
        x_norm = (x - running_mean)/np.sqrt(running_var+eps)
        out = gamma*x_norm + beta
        #cache = (x, x_norm, gamma, beta, eps)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    x, x_norm, sample_mean, sample_var, gamma, eps = cache
    N, D = x.shape

    # add_gate_1(gamma * x_norm, beta)
    dgamma_x_norm = dout
    dbeta = np.sum(dout, axis=0)

    # multiply_gate_1(x_norm, gamma)
    dx_norm = dgamma_x_norm * gamma
    dgamma = np.sum(dgamma_x_norm * x_norm, axis=0)

    # multiply_gate_2(x - mean, 1 / sqrt(var + eps))
    dxu1 = dx_norm * (1 / np.sqrt(sample_var + eps))
    dinv_sqrt_var = np.sum(dx_norm * (x - sample_mean), axis=0)

    # inverse_gate_1(sqrt(var - eps))
    dsqrt_var = -(1 / (sample_var + eps)) * dinv_sqrt_var

    # sqrt_eps_gate_1(var)
    dvar = 0.5 * (1 / np.sqrt(sample_var + eps)) * dsqrt_var

    # sum_gate_1((x - mean)**2)
    dxu_sq = (1. / N) * np.ones(x.shape) * dvar

    # square_gate_1(x - mean)
    dxu2 = 2 * (x - sample_mean) * dxu_sq

    # subtract_gate_1(x, mean)
    dx1 = dxu1 + dxu2
    du = -np.sum(dxu1 + dxu2, axis=0)

    # sum_gate_2(x)
    dx2 = (1. / N) * np.ones(x.shape) * du

    dx = dx1 + dx2


    '''
    mu, x_mu, x_mu_quad, var, sqrt_var, inv_var, norm, norm2, gamma, beta, x, bn_param = cache

    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    #tesp 9
    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    #step 8
    dva2 = dva3*gamma
    dgamma = np.sum(norm*dva3, axis=0)

    #step 7
    dxmu = inv_var*dva2
    dinvar = np.sum(x_mu*dva2, axis=0)

    #step6
    dsqrtvar = -1. /(sqrt_var**2)*dinvar

    #step5
    dvar = 0.5 * (var + eps)**(-0.5)*dsqrtvar

    #step4
    dcare = 1/float(N)*np.ones((x_mu_quad.shape))*dvar

    #step3
    dxmu += 2*x_mu * dcare

    #step2
    dx=dxmu
    dmu = -np.sum(dxmu, axis=0)

    #step1
    dx += 1/float(N)* np.ones(dxmu.shape)*dmu
    '''
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape)<p)/p
        out = x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = None
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)

    N:	Batch	size	(Number	of	images	on	the	4d	tensor)
    F:	Number	of	filters	on	the	convolution	layer
    kW/kH:	Kernel	Width/Height	(Normally	we	use	square	images,	so	kW=kH)
    H/W:	Image	height/width	(Normally	H=W)
    H'/W':	Convolved	image	height/width	(Remains	the	same	as	input	if	proper	padding	is	used)
    Stride:	Number	of	pixels	that	the	convolution	sliding	window	will	travel.
    Padding:	Zeros	added	to	the	border	of	the	image	to	keep	the	input	and	output	size	the	same.
    Depth:	Volume	input	depth	(ie	if	the	input	is	a	RGB	image	depth	will	be	3)
    Output	depth:	Volume	output	depth	(same	as	F)

    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #extract params
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    stride = conv_param['stride']
    padding = conv_param['pad']

    #calculate out layer dimension
    H_OUT = 1 + (H + 2 * padding - HH) / stride
    W_OUT = 1 + (W + 2 * padding - WW) / stride
    h1 = int(H_OUT)
    w1 = int(W_OUT)
    out = np.zeros((N, F, h1, w1))
    #print(out.shape)
    #print('****'*10)
    #image padding
    x_padded = np.lib.pad(x, ((0,0),(0,0),(padding, padding),(padding, padding)), 'constant', constant_values=0)
    #forward step
    for n in range(N): #step over batch image
        #print("sample:%d"%n)
        #print(x_padded[n])
        for f in range(F): #step over filter
            #print("filter:%d"%f)
            #print(w[f])
            for width in range(0,H,stride): #step over width with stride
                #print('width:%d'%width)
                for height in range(0,W,stride): #step over height with stride
                    #print('height%d'%height)
                    h_start = width
                    h_end = h_start+HH

                    w_start = height
                    w_end =w_start+WW

                    h_pos = int(width/stride)
                    w_pos = int(height/stride)
                    #print('----'*4)
                    #print(x_padded[n,:,h_start:h_end, w_start:w_end])
                    #print(x_padded[n,:,h_start:h_end, w_start:w_end] * w[f])
                    #print(w[f])
                    out[n, f, h_pos, w_pos] = np.sum(x_padded[n,:,h_start:h_end, w_start:w_end] * w[f]) + b[f]



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    _, _, H_R, W_R = dout.shape

    stride = conv_param['stride']
    padding = conv_param['pad']

    x_padded = np.lib.pad(x, ((0,0),(0,0),(padding, padding),(padding, padding)), 'constant', constant_values=0)

    #init out
    dx = np.zeros(x_padded.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    print(dout.shape)
    print(x.shape)
    #calculate dx
    for n in range(N):
        for f in range(F):
            for width in range(0,H,stride):
                for height in range(0,W,stride):
                    h_start = width
                    h_end = h_start+HH

                    w_start = height
                    w_end =w_start+WW

                    h_pos = int(width/stride)
                    w_pos = int(height/stride)
                    dx[n, :, h_start:h_end, w_start:w_end] += dout[n,f,h_pos,w_pos]*w[f]

    #print(dx[1])
    #remove padded columns
    delete_rows = [*range(padding)] + [*range(H+padding, H+2*padding)]
    delete_columns =[*range(padding)] + [*range(W+padding, W+2*padding)]
    dx = np.delete(dx, delete_rows, axis=2)
    dx = np.delete(dx,delete_columns,axis=3)
    #print('-------'*5)
    #print(dx[1])

    #calculate dw
    for n in range(N):
        for f in range(F):
            for width in range(H_R):
                for height in range(W_R):
                    h_start = width*stride
                    h_end = h_start+HH

                    w_start = height*stride
                    w_end =w_start+WW

                    h_pos = int(width/stride)
                    w_pos = int(height/stride)
                    dw[f] += dout[n,f,width,height]*x_padded[n,:,h_start:h_end,w_start:w_end]

    #calculate db
    for f in range(F):
        db[f] = np.sum(dout[:,f,:,:])


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data ((N,C,h1,w1))
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    stride = pool_param['stride']
    H_P = pool_param['pool_height']
    W_P = pool_param['pool_width']

    HH = 1 + (H - H_P) / stride
    WW = 1 + (W - W_P) / stride

    h1 = int(HH)
    w1 = int(WW)
    out = np.zeros((N,C,h1,w1))

    for n in range(N):
        for f in range(C):
            for width in range(0,H,stride):
                for height in range(0,W,stride):
                    h_pos = int(width/stride)
                    w_pos = int(height/stride)

                    h_start = width
                    h_end = h_start+H_P

                    w_start = height
                    w_end =w_start+W_P

                    #max elemnt in the window
                    out[n,f,h_pos,w_pos] = np.max(x[n, f, h_start:h_end, w_start:w_end])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    stride = pool_param['stride']
    H_P = pool_param['pool_height']
    W_P = pool_param['pool_width']
    _, _, HH, WW = dout.shape
    #HH, WW = dout.shape

    #inti dx
    dx = np.zeros((N,C,H,W))

    for n in range(N):
        for c in range(C):
            for width in range(HH):
                for height in range(WW):
                    h_start = width*stride
                    h_end = h_start+H_P
                    w_start = height*stride
                    w_end =w_start+W_P

                    x_pool = x[n,c, h_start:h_end, w_start:w_end]
                    #print(x_pool)
                    mask = (x_pool == np.max(x_pool))
                    #print(mask)
                    dx[n,c,h_start:h_end,w_start:w_end] = mask*dout[n,c,width,height]
                    #print(dx[n,c])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    x = x.reshape((C, N*H*W))
    out, cache = batchnorm_forward(x.T, gamma, beta, bn_param)
    out = out.T.reshape((N, C, H, W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.reshape((C,N*W*H))
    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)
    dx = dx.T.reshape((N,C,H,W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
