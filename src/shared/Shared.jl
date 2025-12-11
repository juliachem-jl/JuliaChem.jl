
module Shared
    
    Base.include(@__MODULE__,"./Constants.jl")
    Base.include(@__MODULE__,"./SCFOptions.jl")
    Base.include(@__MODULE__,"./GPUData.jl")
    Base.include(@__MODULE__,"./GPUData_cuda.jl")
    Base.include(@__MODULE__,"./SCFData.jl")
    Base.include(@__MODULE__,"./Indicies.jl")

    Base.include(@__MODULE__,"./JCTiming.jl")
    Base.include(@__MODULE__,"./JCTiming_Setters.jl")
    
   

    global Timing::JCTiming # module singleton variable for timing 

    function reset_timing()
        global Timing = create_jctiming()
    end

    export reset_timing, Timing

end 
