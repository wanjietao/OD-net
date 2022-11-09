def model_fn(features, labels, mode, params):
    
    print('===== training scheme: {} ====='.format(params['FLAGS'].config))
    ## 继承model参数
    params.update(model_params)
    
    if 'new_estimator' in params and params['new_estimator']:
        new_estimator = True
    else:
        new_estimator = False
        
    is_prediction = (new_estimator and tf.estimator.ModeKeys.PREDICT == mode) 
    export = 'export' in params and params['export']
    
    if (new_estimator and mode != tf.estimator.ModeKeys.TRAIN) :
        is_training = False
    else:
        is_training = tf.placeholder_with_default(True, [], name='training') ###?????
        
    features['his_dep_city_code_seq_pad'] = tf.reshape(features['his_dep_city_code_seq_pad'],[-1])
    features['his_arr_city_code_seq_pad'] = tf.reshape(features['his_arr_city_code_seq_pad'],[-1])
    
    
    his_dep_city_code_seq_pad = tf.reshape(tf.string_split(features['his_dep_city_code_seq_pad'], ',', skip_empty=False).values,[-1,32])
    his_arr_city_code_seq_pad = tf.reshape(tf.string_split(features['his_arr_city_code_seq_pad'], ',', skip_empty=False).values,[-1,32])
    
    features['his_dep_city_code_seq_pad'] = his_dep_city_code_seq_pad
    features['his_arr_city_code_seq_pad'] = his_arr_city_code_seq_pad
    
    his_dep_city_code_seq_pad = tf.string_to_hash_bucket_fast(features['his_dep_city_code_seq_pad'], 10000)
    his_arr_city_code_seq_pad = tf.string_to_hash_bucket_fast(features['his_arr_city_code_seq_pad'], 10000)
    
    dep_city_code = tf.string_to_hash_bucket_fast(features['dep_city_code'], 10000)
    arr_city_code = tf.string_to_hash_bucket_fast(features['arr_city_code'], 10000)  
    
    f_level = tf.string_to_hash_bucket_fast(features['f_level'], 1000)
    
    W_city = tf.Variable(tf.random_uniform([10000, 8], -1.0, 1.0))
    W_flevel = tf.Variable(tf.random_uniform([1000, 8], -1.0, 1.0))
    
    his_dep_city_code_seq_emb_temporal = tf.nn.embedding_lookup(W_city, his_dep_city_code_seq_pad)   # [b,32,8]
    his_arr_city_code_seq_emb_temporal = tf.nn.embedding_lookup(W_city, his_arr_city_code_seq_pad)   # [b,32,8]
    
    
    dep_city_code_emb_temporal = tf.nn.embedding_lookup(W_city, dep_city_code)   # [b,1,8]
    arr_city_code_emb_temporal = tf.nn.embedding_lookup(W_city, arr_city_code)   # [b,1,8]
    
    f_level_emb = tf.nn.embedding_lookup(W_flevel, f_level)   # [b,1,8]
    
    ###add spatial embeddings
    features['dep_city_embedding'] = tf.reshape(features['dep_city_embedding'],[-1])
    dep_city_code_emb_spatial = tf.reshape(tf.string_split(features['dep_city_embedding'], ';', skip_empty=False).values,[-1,8]) #[b,8]
    dep_city_code_emb_spatial = tf.reshape(dep_city_code_emb_spatial,[-1,1,8]) # [b,1,8]
    dep_city_code_emb_spatial = tf.string_to_number(dep_city_code_emb_spatial,tf.float32)
    features['arr_city_embedding'] = tf.reshape(features['arr_city_embedding'],[-1])
    arr_city_code_emb_spatial = tf.reshape(tf.string_split(features['arr_city_embedding'], ';', skip_empty=False).values,[-1,8]) #[b,8]
    arr_city_code_emb_spatial = tf.reshape(arr_city_code_emb_spatial,[-1,1,8]) # [b,1,8]
    arr_city_code_emb_spatial = tf.string_to_number(arr_city_code_emb_spatial,tf.float32)
    
    features['his_dep_city_code_embedding_seq_pad'] = tf.reshape(features['his_dep_city_code_embedding_seq_pad'],[-1])
    his_dep_city_code_seq_emb_spatial = tf.reshape(tf.string_split(features['his_dep_city_code_embedding_seq_pad'], ',', skip_empty=False).values,[-1,32])
    his_dep_city_code_seq_emb_spatial = tf.reshape(his_dep_city_code_seq_emb_spatial,[-1])
    his_dep_city_code_seq_emb_spatial = tf.reshape(tf.string_split(his_dep_city_code_seq_emb_spatial, ';', skip_empty=False).values,[-1,32,8])
    his_dep_city_code_seq_emb_spatial = tf.string_to_number(his_dep_city_code_seq_emb_spatial,tf.float32) # [b,32,8]
    
    features['his_arr_city_code_embedding_seq_pad'] = tf.reshape(features['his_arr_city_code_embedding_seq_pad'],[-1])
    his_arr_city_code_seq_emb_spatial = tf.reshape(tf.string_split(features['his_arr_city_code_embedding_seq_pad'], ',', skip_empty=False).values,[-1,32])
    his_arr_city_code_seq_emb_spatial = tf.reshape(his_arr_city_code_seq_emb_spatial,[-1])
    his_arr_city_code_seq_emb_spatial = tf.reshape(tf.string_split(his_arr_city_code_seq_emb_spatial, ';', skip_empty=False).values,[-1,32,8])
    his_arr_city_code_seq_emb_spatial = tf.string_to_number(his_arr_city_code_seq_emb_spatial,tf.float32) # [b,32,8]   
    
    
    
    
    
    # city_code_emb = tf.expand_dims(city_code_emb,1) # [b,1,8]
    # f_level_emb = tf.expand_dims(f_level_emb,1) # [b,1,8]
    
    emb_all = tf.concat([his_dep_city_code_seq_emb_temporal,
                         his_arr_city_code_seq_emb_temporal,
                         dep_city_code_emb_temporal,
                         arr_city_code_emb_temporal,
                         dep_city_code_emb_spatial,
                         arr_city_code_emb_spatial,
                         his_dep_city_code_seq_emb_spatial,
                         his_arr_city_code_seq_emb_spatial,
                         f_level_emb],axis=1) # [b,34,8]
    
    emb_all = tf.reduce_mean(emb_all, axis = 1) # [b,8]
    
    buy_count_history_rate_dep_city = features['buy_count_history_rate_dep_city']
    buy_count_history_rate_arr_city = features['buy_count_history_rate_arr_city']
                        
    query_count_history_rate_dep_city = features['query_count_history_rate_dep_city']
    query_count_history_rate_arr_city = features['query_count_history_rate_arr_city']
                        
    query_count_7day_rate_dep_city = features['query_count_7day_rate_dep_city']
    query_count_7day_rate_arr_city = features['query_count_7day_rate_arr_city']
    uv_visit_norm = features['uv_visit_norm']
    uv_depart_norm = features['uv_depart_norm']
    

    ##############################3 读取features\labels
    model_feature = tf.concat([emb_all,
                               buy_count_history_rate_dep_city,
                               buy_count_history_rate_arr_city,
                               query_count_history_rate_dep_city,
                               query_count_history_rate_arr_city,
                               query_count_7day_rate_dep_city,
                               query_count_7day_rate_arr_city,
                               uv_visit_norm,
                               uv_depart_norm
                               ],axis=-1) #[b,9]
    # model_feature = emb_all
    
    # label_weights = features['label_weight'] 
    # label_weights = tf.reshape(label_weights, [-1, 1]) if not export else 0.0
    
    ## 输入用户偏好占位符
#    self.user_long_bhv = tf.placeholder(tf.float32, [None, max_long_length, user_bhv_size], name="user_long_bhv")
#    self.user_short_bhv = tf.placeholder(tf.float32, [None, max_short_length, user_bhv_size], name="user_short_bhv")
#    self.user_intent_bhv = tf.placeholder(tf.float32, [None, max_intent_length, user_bhv_size], name="user_intent_bhv")
    if is_prediction:
        labels_dep = features['label_dep']  #考虑到预测没有labels, 从feature里获取
        labels_arr = features['label_arr']  #考虑到预测没有labels, 从feature里获取
        print(tf.shape(labels_dep))
        print(tf.shape(labels_arr))
    
    ## model_feature 
    new_inputs = model_feature
    
    ########################### dnn ##################### 
     # res = normal(res,max_len)
    input_state = tf.layers.dense(new_inputs, units=64,   activation=tf.nn.tanh,use_bias=True,name='elu1')
    #dense transformer
    res2 = tf.concat([new_inputs,input_state], axis=1)
    input_state = tf.layers.dense(res2,units=32, activation=tf.nn.tanh,use_bias=True,name='elu3')
    res3 = tf.concat([res2,input_state], axis =1)
    input_state = tf.layers.dense(res3,units=16, activation=tf.nn.tanh,use_bias=True,name='elu4')
    res4 = tf.concat([res3,input_state], axis =1)
    input_state = tf.layers.dense(res4,units=4, activation=tf.nn.tanh,use_bias=True,name='elu5')
    res5 = tf.concat([res4,input_state], axis =1)
    logits_dep = tf.contrib.layers.fully_connected(res5, 1, activation_fn=None)
    logits_arr = tf.contrib.layers.fully_connected(res5, 1, activation_fn=None)
    
    preds_dep = tf.nn.sigmoid(logits_dep)
    preds_arr = tf.nn.sigmoid(logits_arr)
    
    rank_predict_dep = tf.identity(preds_dep, name='rank_predict_dep')
    rank_predict_arr = tf.identity(preds_arr, name='rank_predict_arr')
    
    m_rank_predict_dep = tf.reshape(rank_predict_dep, [-1, ])
    m_rank_predict_arr = tf.reshape(rank_predict_arr, [-1, ])
    predictions = {
        # 'user_id':features['user_id'],
        # 'pay_time_new':features['pay_time_new'],
        # 'dep_time':features['dep_time'],
        # 'country_code':features['country_code'],
        'score_dep': m_rank_predict_dep,
        'score_arr': m_rank_predict_arr
    }
    
    # output = tf.layers.dense(res5,units=params["FLAGS"].n_class, activation=None,use_bias=True,name='linear6')

    

    #decoder and attention
    # rank_predict = tf.convert_to_tensor(output, name="Output")
    
    # predictions = {
    #     'score': rank_predict
    # }
    export_outputs = {
        'score': tf.estimator.export.PredictOutput(predictions)
    }
    if is_prediction:
        evaluation_fields = {'labels_dep': tf.reshape(labels_dep, [-1,1]),
                             'labels_arr': tf.reshape(labels_arr, [-1,1])
                             }
    else:
        evaluation_fields = {}
    
    ############################################预测时输出#########################3
    if tf.estimator.ModeKeys.PREDICT == mode:
        predictions.update(evaluation_fields)
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs)

    labels_dep = tf.reshape(labels['labels_dep'], [-1, 1])
    labels_dep = tf.cast(labels_dep, dtype=tf.float32)
    labels_arr = tf.reshape(labels['labels_arr'], [-1, 1])
    labels_arr = tf.cast(labels_arr, dtype=tf.float32)
    ##### loss function 
    # loss = 0.0
    with tf.name_scope("loss_dep"):        
        loss_dep = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_dep,
                logits=logits_dep)
        )
        auc_dep = tf.metrics.auc(labels_dep, preds_dep)
        
    with tf.name_scope("loss_arr"):        
        loss_arr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_arr,
                logits=logits_arr)
        )
        auc_arr = tf.metrics.auc(labels_arr, preds_arr)   
    print('auc_dep: ', auc_dep)

    auc = 0.5*auc_dep[1]+0.5*auc_arr[1]
    
    #####################################以下代码暂不修改################################
    # 5500w sample, 1.1w steps per epoch with batch_size=5k
    # 0.90 decay [0.59, 0.35, 0.21, 0.12, 0.07, 0.04, 0.03, 0.01, ...]
    # 0.85 decay [0.44, 0.20, 0.09, 0.04, 0.02, 0.01, 0.003, 0.002, ...]
    # 0.80 decay [0.33, 0.11, 0.04, 0.01, 0.004, ...]
    lr = 1e-4 if 'lr' not in params else float(params['lr'])
 
    learning_rate = tf.cond(
        tf.train.get_global_step() <= 1000000,
        lambda: lr,
        lambda: tf.train.exponential_decay(lr, tf.train.get_global_step(), 1000000, 0.95, staircase=True)
    )

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss_dep', loss_dep)
    tf.summary.scalar('auc_dep', auc_dep[1])
    tf.summary.scalar('loss_arr', loss_arr)
    tf.summary.scalar('auc_arr', auc_arr[1]) 
    tf.summary.scalar('auc_arr', auc) 
    # is_chief=False
    # worker_count =1

            
    if params["FLAGS"].train_syn:
        is_chief=False
        worker_count =1
        print("syn")
        if 'TF_CONFIG' in os.environ:
            CONFIG = json.loads(os.environ['TF_CONFIG'])
            print(CONFIG)
            if 'worker' in CONFIG['cluster']:
                worker_count = len(CONFIG['cluster']['worker'])+len(CONFIG['cluster']['chief'])
                
            if CONFIG['task']['type'] == 'chief':
                is_chief = True
            else:
                is_chief = False
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # SyncReplicasOptimizer
        optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=worker_count, total_num_replicas=worker_count)
    else:
        print("asyn")
        optimizer = tf.train.AdamAsyncOptimizer(learning_rate=learning_rate) #阿里自研
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss = 0.5*loss_dep+0.5*loss_arr
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn layers
#    with tf.control_dependencies(update_ops):
#        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        
    
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=training_hooks)
#    if mode == tf.estimator.ModeKeys.EVAL:
#        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric_fns)
    
    
    metric_train_fns = {"loss": loss,
                        "loss_dep": loss_dep,
                        "loss_arr": loss_arr,
                        "auc": auc,
                        "auc_dep": auc_dep[1],
                        "auc_arr": auc_arr[1],
                   'step': tf.train.get_global_step(),
                   'learning_rate': learning_rate
                   }
    # metric_fns ={}
    # extra_metrics ={}
    # for k in [1, 2]:
    #     correct = tf.nn.in_top_k(output, tf.argmax(tf.cast(labels,tf.float32),1), k)
    #     print(correct)
    #     extra_metrics["acc@" + str(k)] = tf.metrics.accuracy(
    #         labels=tf.ones_like(tf.argmax(labels,1), dtype=tf.float32),
    #         predictions=tf.to_float(correct))
    # metric_fns.update({
    #         "metric/recall_top@%d" % topn: tf.metrics.recall_at_k(
    #             tf.argmax(tf.cast(labels,tf.float32),1), output, topn) for topn in [1, 2]
    #     })
    # metric_fns.update(extra_metrics)
    
    preds_bool_dep = tf.greater(preds_dep, tf.ones_like(preds_dep) * 0.5)
    preds_bool_arr = tf.greater(preds_arr, tf.ones_like(preds_arr) * 0.5)
    
    metric_fns = {'recall_dep': tf.metrics.recall(labels_dep, preds_bool_dep),
                  'recall_arr': tf.metrics.recall(labels_arr, preds_bool_arr),
                  
                     'precision_dep': tf.metrics.precision(labels_dep, preds_bool_dep), 
                     'precision_arr': tf.metrics.precision(labels_arr, preds_bool_arr),
                     
                     'auc_dep': tf.metrics.auc(labels_dep, preds_dep),
                     'auc_arr': tf.metrics.auc(labels_arr, preds_arr),
                     }
    
    
    logging_hook = tf.train.LoggingTensorHook(metric_train_fns, every_n_iter=50)

    # logging_hook = LogviewMetricHook(metric_train_fns,tf.train.get_global_step(),logviewMetricWriter)

    if params["FLAGS"].train_syn:
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief,num_tokens=-1)   
        training_hooks = [logging_hook,sync_replicas_hook]
    else:  
        training_hooks = [logging_hook]
    evaluation_hooks=[LogviewMetricHook(metric_fns, tf.train.get_global_step(), logviewMetricWriter)]
    
    # change
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     estimator_spec = tf.estimator.EstimatorSpec(
    #         mode=tf.estimator.ModeKeys.TRAIN,
    #         loss=loss,
    #         train_op=train_op
    #     )
    # elif mode == tf.estimator.ModeKeys.EVAL:
    #     estimator_spec = tf.estimator.EstimatorSpec(
    #         mode=tf.estimator.ModeKeys.EVAL,
    #         loss=loss,
    #         eval_metric_ops=metric_fns,
    #         evaluation_hooks=evaluation_hooks
    #     )
    # return estimator_spec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metric_fns,
        evaluation_hooks=evaluation_hooks,
        training_hooks = training_hooks
    )
