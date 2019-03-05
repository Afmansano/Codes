import tensorflow as tf 
import numpy as np
from tensorflow.contrib.layers import fully_connected, dropout, batch_norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

he_init = tf.contrib.layers.variance_scaling_initializer()

class DNNClassifier(BaseEstimator):

    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.001, batch_size=20, activation=tf.nn.elu,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None):
        self.n_hidden_layers= n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    
    def _dnn(self, inputs):

        normalization = batch_norm if self.batch_norm_momentum else None
        bn_params = {
            "is_training" : self._training,
            "decay": self.batch_norm_momentum,
            "update_collections": None
        }
        for layer in range (self.n_hidden_layers):
            if self.dropout_rate:
                inputs = dropout(inputs, (1.0 - self.dropout_rate), self._training)
            inputs = fully_connected(inputs, self.n_neurons, activation_fn=self.activation, 
                                    normalizer_fn=normalization, normalizer_params=bn_params,
                                    weights_initializer=he_init, scope="hidden%d" % (layer+1))
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        self._training = True if self.batch_norm_momentum or self.dropout_rate else None
        dnn_outputs = self._dnn(X)
        logits = fully_connected(dnn_outputs, n_outputs, activation_fn=None, weights_initializer=he_init, scope="logits")
        y_proba = tf.nn.softmax(logits, name="y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #facilitar o acesso a operacoes importantes
        self._X, self._y = X, y
        self._y_proba = y_proba
        self._loss = loss
        self._training_op = training_op
        self._accuracy = accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return{gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name+"/Assign")
                      for gvar_name in gvar_names}
        init_values =  {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)
        
    def fit(self, X, y, n_epochs=1000, X_valid=None, y_valid=None):
        self.close_session()
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        #cria um mapeamento para o indice do atributo classes_
        self.class_to_index_ = {label: index 
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        max_chekcs_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        #training
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy],
                                                 feed_dict={self._X: X_valid,
                                                            self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress+=1
                        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                               epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > max_chekcs_without_progress:
                        print("Early stopping!")
                        break
                else:
                    loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                        feed_dict={self._X: X_batch,
                                                                    self._y: y_batch})
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                            epoch, loss_train, acc_train * 100))
                if best_params:
                    self._restore_model_params(best_params)
            return self    

        
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % sell.__class__.__name__)
        with self._session.as_default() as sess:
            return self._y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                            for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)

         

