import tensorflow as tf
import model

def model_fn(features,labels,mode,params):
    batch_size = params['batch_size']
    embedding_size = params['embedding_size']
    attention_size = params['attention_size']
    hidden_size = params['hidden_size']
    n_mid = params['n_mid']
    n_uid = params['n_uid']
    n_cat = params['n_cat']
    
    DIN_model = model.DIN(n_mid=n_mid,n_uid=n_uid,n_cat=n_cat,
                    HIDDEN_SIZE=hidden_size,ATTENTION_SIZE=attention_size,
                    EMBEDDING_DIM=embedding_size)
    
    with tf.name_scope('model'):
        output = DIN_model.forward(features)
        prob = output

    if mode == tf.estimator.ModeKeys.PREDICT: # PREDICT mode
        print("enter tf.estimator.ModeKeys.PREDICT")
        return tf.estimator.EstimatorSpec(mode, predictions=
                {'probabilities': prob})

    tf.add_to_collection('prob', prob)
    auc = tf.metrics.auc(tf.expand_dims(labels, -1),
            tf.expand_dims(prob,-1), num_thresholds=batch_size)
    tf.summary.scalar('accuracy', auc[1])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=tf.cast(labels,dtype=tf.float32)))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'accuracy':auc})
    
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    global_step = tf.train.get_or_create_global_step()      
    with tf.name_scope('compute_apply'):
        grad_vars = optimizer.compute_gradients(loss)
        #clip_gradient, _ = tf.clip_by_global_norm(grad_vars, 5.0)
        apply_op = optimizer.apply_gradients(grad_vars, global_step)
        tf.add_to_collection('auc', auc[1])
        
    train_op = apply_op
    return tf.estimator.EstimatorSpec(mode,
                loss=loss,
                train_op=train_op)
                #training_hooks=[logging_hooks]
    


