# sourcery skip: hoist-statement-from-loop
import subprocess
import config
api_key = config.API_KEY
basedir = "/homes/fabadmus/Internship/grad_project/"
out_dir = basedir + "arthritis/"

# parkinsons as target and arthritis as control
target = 'TWDIS_06685'
control = 'TWDIS_09015'


for interm in ['PATH', 'HGNC', 'TWMET']:
    for mysize in ['10', '25', '50']:
        for sl_size in ['2', '4']:
            target_out_file  = out_dir + interm + '_' + target  + '_' + mysize + '.flt.txt'
            control_out_file = out_dir + interm + '_' + control + '_' + mysize + '.flc.txt'
            label_out_file = out_dir + interm + '_' + target + '_' + control + '_'  + sl_size + '_' + 'labelfile' + '.txt'
            sl_out_file = out_dir + interm + '_' + target + '_' + control + '_'  + sl_size + '_' + 'slfile' + '.txt'
            emb_out_file = out_dir + interm + '_' + target + '_' + control + '_'  + mysize + '_' + sl_size + '_' + 'embedding' + '.emb'
            pred_out_file = out_dir + interm + '_' + target + '_' + control + '_'  + mysize + '_' + sl_size + '_' + 'predictions' + '.txt'
            
            
            # First we create the first layer for the target and the control
            my_targ_cmd =  "python " + basedir +"get_first_layer.py"   + " -p " + interm + " -i " + target  + " -n " + mysize + " -o " + target_out_file
            my_cont_cmd =  "python " + basedir +"get_first_layer.py"   + " -p " + interm + " -i " + control + " -n " + mysize + " -o " + control_out_file
            
            # Now we add the connecting intermediates. These are always metabolites for now. This can be changed later.
            my_sl_cmd  =  "python " + basedir +"get_second_layer.py"  + " -t " + target_out_file + " -c " + control_out_file + " -l " + label_out_file + " -o " + sl_out_file + " -s " + sl_size
            
            # now we embedd the networks
            my_emb_cmd = "python " + basedir +"get_embedding.py" + " -s " + sl_out_file + " -o " + emb_out_file
            
            # Now we train and predict 
            my_pred_cmd = "python " + basedir + "run_model.py" + " -s " + sl_out_file + " -m " + label_out_file + " -e " + emb_out_file + " -a " + "randomforest" + " -o " + pred_out_file
            
            
            subprocess.run(my_targ_cmd, shell=True)
            subprocess.run(my_cont_cmd, shell=True)
            subprocess.run(my_sl_cmd, shell=True)
            subprocess.run(my_emb_cmd, shell=True)
            subprocess.run(my_pred_cmd, shell=True)


