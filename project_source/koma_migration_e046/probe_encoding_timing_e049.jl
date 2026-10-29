using KomaMRI, HDF5, TOML
seq=read_seq(ARGS[1];verbose=false)
b=findfirst(a->a.N>0,seq.ADC)
r=findlast(i->any(!iszero,seq.RF[1,i].A),1:b)
shot=copy(seq[r:b]);actual=get_adc_sampling_times(shot)
expected=h5open(f->read(f,"adc_time_s"),ARGS[2],"r")
report=Dict("actual_first_s"=>actual[1],"exported_first_s"=>expected[1],
            "constant_difference_s"=>actual[1]-expected[1],"dwell_s"=>dwell(shot.ADC[end]),
            "maximum_delta_s"=>maximum(abs.(actual.-expected)),"duration_s"=>dur(shot))
TOML.print(stdout,report)
