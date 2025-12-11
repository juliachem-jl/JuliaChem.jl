j = 0
t = @time begin
    for i in 1:10000
        j = i
    end
end
println(t)