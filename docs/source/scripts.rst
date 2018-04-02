=======
Scripts
=======

Standalone Processing Examples
==============================

Using Scaler Train and Test Helper
----------------------------------

Train a DNN using the Scaler-Normalized AntiNex Django Dataset. This builds the train and test datasets using the ``build_scaler_train_and_test_datasets`` method from the internal modules.

.. automodule:: antinex_core.scripts.antinex-scaler-django
   :members: dataset,model_backup_file,model_weights_file,model_json_file,model_image_file,footnote_text,image_title,show_predictions,features_to_process,min_scaler_range,max_scaler_range,test_size,batch_size,epochs,num_splits,loss,optimizer,metrics,histories,label_rules,build_model,model,scaler_res,train_scaler_df,sample_rows,kfold,results,scores,cm,sample_predictions,merged_predictions_df,predict_rows_df,fig,ax

Using Manual Scaler Objects
---------------------------

Train a DNN using the Scaler-Normalized AntiNex Django Dataset. This builds the train and test datasets manually to verify the process before editing the ``build_scaler_train_and_test_datasets`` method.

.. automodule:: antinex_core.scripts.standalone-scaler-django
   :members: dataset,model_backup_file,model_weights_file,model_json_file,model_image_file,datanode,footnote_text,image_title,show_predictions,features_to_process,min_scaler_range,max_scaler_range,test_size,batch_size,epochs,num_splits,loss,optimizer,metrics,histories,label_rules,build_model,model,scaler_res,train_scaler_df,sample_rows,kfold,results,scores,cm,sample_predictions,merged_predictions_df,predict_rows_df,test_df,train_df,test_scaler,train_scaler,train_only_floats,test_only_floats,fig,ax

Convert Bottom Rows from a CSV File into JSON
=============================================

When testing live DNN predictions you can use this utility script to print a few JSON-ready dictionaries out to ``stdout``. 

Usage:

::

    convert-bottom-rows-to-json.py -f <CSV File> -b <Optional - number of rows from the bottom>

.. automodule:: antinex_core.scripts.convert-bottom-rows-to-json
   :members: parser,dataset,bottom_row_idx,output_predict_rows
