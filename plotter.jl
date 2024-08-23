using CSV;
using DataFrames;
using Plots;
plotly();

linewidth=4;

filename = "Data/exhaustive_sweep.csv";
df = CSV.read(filename, DataFrame; header=1);
other_file = "Data/exhaustive_sweep_back.csv";
other_df = CSV.read(other_file, DataFrame; header=1);
df = vcat(df, other_df);
df = unique(df, [:t1,:t2,:vx,:detune_ratio]); 

df.t1us = ceil.(df.t1);
df.t1 = 1e-6.*(df.t1);
df.t2us = ceil.(df.t2);
df.t2 = 1e-6.*(df.t2);

single_shot_df = copy(df);
single_shot_df.argmax_t = single_shot_df.argmax_t_vy;;
single_shot_df.signal = single_shot_df.max_vy; # Set the signal column
single_shot_df.improvement = single_shot_df.signal ./ single_shot_df.max_vy_r; # Set the improvement column

multiple_shot_df = copy(df);
multiple_shot_df.argmax_t = multiple_shot_df.argmax_t_vyst;
multiple_shot_df.signal = multiple_shot_df.max_vyst; # Set the signal column
multiple_shot_df.improvement = multiple_shot_df.signal ./ multiple_shot_df.max_vyst_r; # Set the improvement column

# # Load the single shot (= optimise max vy) dataset
# filename = "finer_resolution.csv";
# single_shot_df = CSV.read(filename, DataFrame; header=1);
# other_file = "smaller_t1.csv";
# other_single_shot_df = CSV.read(other_file, DataFrame; header=1);
# # single_shot_df = other_single_shot_df;
# single_shot_df = vcat(single_shot_df, other_single_shot_df);
# single_shot_df = unique(single_shot_df, [:t1,:t2,:vx,:detune_ratio]); 

# single_shot_df.argmax_t = single_shot_df.argmax_t_vy;;
# single_shot_df.signal = single_shot_df.max_vy; # Set the signal column
# single_shot_df.improvement = single_shot_df.signal ./ single_shot_df.max_vy_r; # Set the improvement column
# # T1 and T2 are stored in units of us. Convert to integers and
# # store in a new column.
# single_shot_df.t1us = ceil.(single_shot_df.t1);
# single_shot_df.t1 = 1e-6.*(single_shot_df.t1);
# single_shot_df.t2us = ceil.(single_shot_df.t2);
# single_shot_df.t2 = 1e-6.*(single_shot_df.t2);

# # Load the multiple shot (= optimise max vy/sqrt(t)) dataset
# filename = "new_vyst.csv";
# multiple_shot_df = CSV.read(filename, DataFrame; header=1);

# multiple_shot_df.argmax_t = multiple_shot_df.argmax_t_vyst;
# multiple_shot_df.signal = multiple_shot_df.max_vyst; # Set the signal column
# multiple_shot_df.improvement = multiple_shot_df.signal ./ multiple_shot_df.max_vyst_r; # Set the improvement column
# # T1 and T2 are stored in units of us. Convert to integers and
# # store in a new column.
# multiple_shot_df.t1us = ceil.(multiple_shot_df.t1);
# multiple_shot_df.t1 = 1e-6.*(multiple_shot_df.t1);
# multiple_shot_df.t2us = ceil.(multiple_shot_df.t2);
# multiple_shot_df.t2 = 1e-6.*(multiple_shot_df.t2);

function logrange(start,stop,len)
    @assert stop > start;
    @assert len > 1;
    stepmul = (stop/start) ^ (1/(len-1));
    return start * (stepmul .^ (0:len-1));
end

function filter_t1(df, t1_value_choice)
    # Decide whether we want to plot for a single value for T1 or for all T1 values.
    if t1_value_choice == 1
        # Pick out the T1 corresponding to min T1 greater than T2.
        t1_value = minimum(filter([:t1us] => x -> x > df.t2us[1], df).t1us);
        df = filter([:t1us] => x -> abs(x-t1_value) < 1e-1, df);
    elseif t1_value_choice == 2
        # Pick out the T1 corresponding to max T1 lesser than T2.
        t1_value = maximum(filter([:t1us] => x -> x < df.t2us[1], df).t1us);
        df = filter([:t1us] => x -> abs(x-t1_value) < 1e-1, df);
    elseif t1_value_choice == 3
        # Pick out the T1 corresponding to the custom value.
        t1_value = 109;
        df = filter([:t1us] => x -> abs(x-t1_value) < 1e-1, df);
    end
    return df;
end

function create_heatmap_data(x,y,z)
    xs = sort(unique(x));
    ys = sort(unique(y));
    n = length(xs);
    m = length(ys);
    A = zeros((m, n));
    D1 = Dict(x => i for (i,x) in enumerate(xs));
    D2 = Dict(x => i for (i,x) in enumerate(ys));
    for i in 1:size(z, 1)
        A[D2[y[i]], D1[x[i]]] = z[i];
    end
    return (xs, ys, A);
end

function smooth_heatmap_using_average(xs, zs, vx_list)
    D1 = Dict(x => i for (i,x) in enumerate(xs));
    for vx in vx_list
        # Find the matching element
        difference = Inf;
        match = 0;
        for i in xs
            if abs(i - vx) < difference
                difference = abs(i - vx);
                match = i;
            else
                break;
            end
        end
        zs[:,D1[match]] = 0.6*zs[:,D1[match]-1]+0.4*zs[:,D1[match]+1];
    end
end

function smooth_heatmap_using_left(xs, ys, zs)
    D1 = Dict(x => i for (i,x) in enumerate(xs));
    D2 = Dict(y => i for (i,y) in enumerate(ys));
    for i in 1:(size(xs,1)-1)
        for j in 1:size(ys,1)
            if abs(zs[j,i] - zs[j,i+1])>0.1
                zs[j,i+1] = zs[j,i]
            end
        end
    end
end

function get_max_deviation_from_1(zs)
    return maximum(zs) + minimum(zs) > 1 ? maximum(zs)-1 : 1-minimum(zs);
end

function improvement_ratio_for_detuning(vx,detune_ratio,df)
    buff_df = filter([:vx,:detune_ratio]=>(a,b)->a≈vx && b≈detune_ratio, df)[1,:];
    return buff_df.improvement;
end

function

# Decide which df to use
df = multiple_shot_df;
choice = "multiple";
df = single_shot_df;
choice = "single";

# Figure 1
# Plot max vy as a function of initial state and detuning for a fixed T1/T2 (heatmap).
buff_df = filter([:t1us] => x -> x == 61 || x == 101 || x == 100, df);
# Start the actual plot
for temp_df in groupby(buff_df, [:t1us])
    new_df = temp_df;
    graph = plot();
    max_diff = get_max_deviation_from_1(new_df.improvement);
    xs, ys, zs = create_heatmap_data(asin.(new_df.vx) ./ pi, new_df.detune_ratio, new_df.improvement);
    if choice == "single"
        smooth_heatmap_using_average(xs, zs, [0.26,0.46,0.47]);
    else
        smooth_heatmap_using_average(xs, zs, [0.46,0.47]);
    end
    # stable_threshold_vx = 0.5 * sqrt(temp_df.t2us[1]/temp_df.t1us[1]);
    plot!(graph, xs, ys, zs,
            title="Improvement ratio = f(vx, detuning) (T1/T2 = $(new_df.t1us[1]/new_df.t2us[1]))",
            xlabel="Initial state (theta)", ylabel="Detuning/gamma2",
            label="(SNR for Coherence stabilisation))/(SNR for Ramsey)",
            st=:heatmap, interpolate=false, c=:seismic, clims=(1-max_diff, 1+max_diff),
            # xrotation = 90, yrotation = 45,
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm, topmargin=10mm);
    display(graph);
    savefig(graph, "$(choice) - improvement vs vx, detuning (T1:T2=$(new_df.t1us[1]/new_df.t2us[1])).svg")
end

# Figure 2
# Plot max vy as a function of initial state and T1/T2 for a fixed detuning/gamma2 = 0.01 (heatmap).
# # Need to use a new file
# filename = "Data/improvement_vs_t1_vx_sweep.csv";
# temp_df = CSV.read(filename, DataFrame; header=1);

# temp_df.improvement = temp_df.max_vy ./ temp_df.max_vy_r; # Set the improvement column
# # T1 and T2 are stored in units of us. Convert to integers and
# # store in a new column.
# temp_df.t1us = ceil.(temp_df.t1);
# temp_df.t1 = 1e-6.*(temp_df.t1);
# temp_df.t2us = ceil.(temp_df.t2);
# temp_df.t2 = 1e-6.*(temp_df.t2);

buff_df = filter([:detune_ratio,:t1us] => (x,y) -> x == 0.01 && y < 500, df);
# Start the actual plot
for temp_df in groupby(buff_df, [:detune_ratio])
    new_df = temp_df;
    graph = plot();
    max_diff = get_max_deviation_from_1(new_df.improvement);
    xs, ys, zs = create_heatmap_data(asin.(new_df.vx) ./ pi, new_df.t1us ./ new_df.t2us, new_df.improvement);
    smooth_heatmap_using_left(xs, ys, zs);
    plot!(graph, xs, ys, zs,
            title="Improvement ratio = f(vx, T1/T2) (Detuning/gamma2 = $(new_df.detune_ratio[1]))",
            xlabel="Initial state (theta)", ylabel="T1/T2",
            label="(SNR for Coherence stabilisation))/(SNR for Ramsey)",
            st=:heatmap, interpolate=false, c=:seismic, clims=(1-max_diff, 1+max_diff),
            # xrotation = 90, yrotation = 45,
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm);
    display(graph);
    savefig(graph, "$(choice) - improvement vs vx, T1:T2 (Detuning = $(new_df.detune_ratio[1])*gamma2).svg")
end

# Figure 3
# Max improvement as a function of T1/T2, Detuning
# https://stackoverflow.com/questions/65024962/select-rows-of-a-dataframe-containing-minimum-of-grouping-variable-in-julia
max_filter_df = combine(sdf -> sdf[argmax(sdf.improvement), :], groupby(df, [:t1,:t2,:detune_ratio]));
buff_df = filter([:t1us] => (y) ->y <=200, max_filter_df);
max_diff = get_max_deviation_from_1(buff_df.improvement);
graph = plot();
xs, ys, zs = create_heatmap_data(buff_df.detune_ratio, buff_df.t1us ./ buff_df.t2us, buff_df.improvement);
# stable_threshold_vx = 0.5 * sqrt(temp_df.t2us[1]/temp_df.t1us[1]);
plot!(graph, xs, ys, zs,
        title="Max Improvement ratio = f(Detuning, T1/T2)",
        xlabel="Detuning/gamma2", ylabel="T1/T2",
        label="(SNR for Coherence stabilisation))/(SNR for Ramsey)",
        st=:heatmap, interpolate=true, c=:seismic, clims=(1-max_diff, 1+max_diff),
        # xrotation = 90, yrotation = 45,
        grid=true, gridlinewidth=3, gridstyle=:solid,
        size=(1500,800), linewidth=linewidth,
        leftmargin=10mm, bottommargin=10mm);
# plot!(graph, [stable_threshold_vx], label="Stable threshold", linewidth=linewidth, seriestype=:vline);
display(graph);
savefig(graph, "$(choice) - max improvement vs T1:T2, detuning.svg")

# Figure 4
# Max improvement as a function of T1/T2 for a fixed detuning/gamma
max_filter_df = combine(sdf -> sdf[argmax(sdf.improvement), :], groupby(df, [:t1,:t2,:detune_ratio]));
buff_df = filter([:detune_ratio] => (y) -> y == 0.01, max_filter_df);
# Start the actual plot
for temp_df in groupby(buff_df, [:detune_ratio])
    sort!(temp_df, [:t1us])
    graph = plot();
    xs = temp_df.t1us./temp_df.t2us;
    plot!(graph, xs, temp_df.improvement,
            title="Max Improvement ratio  vs T1/T2 (detuning/gamma2 = $(temp_df.detune_ratio[1]))",
            xlabel="T1/T2", ylabel="Improvement ratio",
            label="(SNR for Coherence stabilisation))/(SNR for Ramsey)",
            xlims=(0.5,maximum(xs)),
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm);
    display(graph);
    savefig(graph, "$(choice) - max improvement vs T1:T2 (Detuning=$(temp_df.detune_ratio[1])*gamma2).svg")
end

# Figure 5
# Argmax state as a function of T1/T2 for a fixed detuning/gamma (=0.1)
max_filter_df = combine(sdf -> sdf[argmax(sdf.improvement), :], groupby(df, [:t1,:t2,:detune_ratio]));
buff_df = filter([:detune_ratio] => (y) -> y == 0.01, max_filter_df);
# Start the actual plot
for temp_df in groupby(buff_df, [:detune_ratio])
    sort!(temp_df, [:t1us])
    graph = plot();
    xs = temp_df.t1us./temp_df.t2us;
    plot!(graph, xs, asin.(temp_df.vx) ./ pi,
            title="Optimal initial state  vs T1/T2 (detuning/gamma2 = $(temp_df.detune_ratio[1]))",
            xlabel="T1/T2", ylabel="Initial state (theta)",
            label="Optimal initial state",legend=:right,
            xlims=(0.5,maximum(xs)), ylims=(0.1,0.35),
            markershape=:square, markercolor=:blue, linecolor=:blue,
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm, rightmargin=20mm);
    # This does not work for plotly() backend
    plot!(twinx(), xs, temp_df.argmax_t ./ temp_df.t2,
            markershape=:circle, markercolor=:red, linecolor=:red, linewidth=linewidth,
            ylabel="Time to achieve max (units of T2)",label="Time to achieve max", rightmargin=20mm,leftmargin=10mm)
    display(graph);
    # savefig(graph, "$(choice) - optimal initial state vs T1:T2 (Detuning=$(temp_df.detune_ratio[1])*gamma2).svg")
end

# Figure
# maximising time vs initial state
buff_df = filter([:t1us, :detune_ratio] => (x, y) -> x==100 && y == 0.01, df);
# Start the actual plot
for temp_df in groupby(buff_df, [:detune_ratio, :t1us])
    sort!(temp_df, [:vx])
    graph = plot();
    xs = asin.(temp_df.vx) ./ pi;
    plot!(graph, xs, temp_df.argmax_t,
            title="Maximising intial time (T1/T2 = $(temp_df.t1us[1] ./ temp_df.t2us[1]), detuning/gamma2 = $(temp_df.detune_ratio[1]))",
            xlabel="Initial state (theta)", ylabel="Maximising time (units of T2)",
            label="Argmax t",
            markershape=:square, markercolor=:blue, linecolor=:blue,
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm, rightmargin=20mm);
    display(graph);
    savefig(graph, "$(choice) - optimal time vs initial state (T1:T2=$(temp_df.t1us[1] ./ temp_df.t2us[1]), Detuning=$(temp_df.detune_ratio[1])*gamma2).svg")
end

# Figure 6
# Signal vs detuning for a fixed vx, T1/T2
max_filter_df = combine(sdf -> sdf[argmax(sdf.improvement), :], groupby(df, [:t1,:t2,:detune_ratio]));
buff_df = filter([:t1us] => x -> x == 61 || x == 109 || x == 100, max_filter_df);
# Start the actual plot
for temp_df in groupby(buff_df, [:t1us])
    sort!(temp_df, [:detune_ratio])
    graph = plot();
    plot!(graph, temp_df.detune_ratio, temp_df.signal,
            title="Signal vs detuning (T1/T2=$(temp_df.t1us[1]/temp_df.t2us[1]))",
            xlabel="Detuning/gamma2", ylabel="SNR",
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm);
    display(graph)
    savefig(graph, "$(choice) - signal vs detuning (T1:T2=$(temp_df.t1us[1]/temp_df.t2us[1])).svg")
end

# Figure 7
# Signal vs detuning for a fixed vx, T1/T2
max_filter_df = combine(sdf -> sdf[argmax(sdf.improvement), :], groupby(df, [:t1,:t2,:detune_ratio]));
buff_df = filter([:t1us] => x -> x == 61 || x == 109 || x == 100, max_filter_df);
# Start the actual plot
for temp_df in groupby(buff_df, [:t1us])
    sort!(temp_df, [:detune_ratio])
    graph = plot();
    plot!(graph, temp_df.detune_ratio, temp_df.improvement,
            title="Signal vs detuning (T1/T2=$(temp_df.t1us[1]/temp_df.t2us[1]))",
            xlabel="Detuning/gamma2", ylabel="Improvement ratio",
            grid=true, gridlinewidth=3, gridstyle=:solid,
            size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm);
    display(graph)
    savefig(graph, "$(choice) - improvement vs detuning (T1:T2=$(temp_df.t1us[1]/temp_df.t2us[1])).svg")
end

# Misclibration plots
# Load the vy dataset
# filename = "miscalibration.csv"; 
# other_file = "miscalibration_to_20.csv";
# single_shot_miscalibration_df = vcat(single_shot_miscalibration_df, CSV.read(other_file, DataFrame; header=1));
# single_shot_miscalibration_df = unique(single_shot_miscalibration_df, [:t1,:t2,:vx,:detune_ratio,:gamma1_miscalib_percent, :gamma2_miscalib_percent]);
filename = "Data/v_miscalibration.csv";
single_shot_miscalibration_df = CSV.read(filename, DataFrame; header=1);

# T1 and T2 are stored in units of us. Convert to integers and
# store in a new column.
single_shot_miscalibration_df.t1us = ceil.(single_shot_miscalibration_df.t1);
single_shot_miscalibration_df.t1 = 1e-6.*(single_shot_miscalibration_df.t1);
single_shot_miscalibration_df.t2us = ceil.(single_shot_miscalibration_df.t2);
single_shot_miscalibration_df.t2 = 1e-6.*(single_shot_miscalibration_df.t2);

single_shot_miscalibration_df.arg_t_r = single_shot_miscalibration_df.arg_t_vy_r;
single_shot_miscalibration_df.arg_t = single_shot_miscalibration_df.arg_t_vy;
single_shot_miscalibration_df.ramsey_miscalib = @. exp(-single_shot_miscalibration_df.arg_t_r * (1+single_shot_miscalibration_df.gamma2_miscalib_percent/100)/single_shot_miscalibration_df.t2) * sin(2*pi*single_shot_miscalibration_df.detune_ratio * single_shot_miscalibration_df.arg_t_r / single_shot_miscalibration_df.t2);
single_shot_miscalibration_df.improvement = single_shot_miscalibration_df.vy ./ single_shot_miscalibration_df.ramsey_miscalib;

# Load the vy/sqrt(t) dataset
filename = "Data/v_sqrt_t_miscalibration.csv";
multiple_shot_miscalibration_df = CSV.read(filename, DataFrame; header=1);

# T1 and T2 are stored in units of us. Convert to integers and
# store in a new column.
multiple_shot_miscalibration_df.t1us = ceil.(multiple_shot_miscalibration_df.t1);
multiple_shot_miscalibration_df.t1 = 1e-6.*(multiple_shot_miscalibration_df.t1);
multiple_shot_miscalibration_df.t2us = ceil.(multiple_shot_miscalibration_df.t2);
multiple_shot_miscalibration_df.t2 = 1e-6.*(multiple_shot_miscalibration_df.t2);

multiple_shot_miscalibration_df.arg_t_r = multiple_shot_miscalibration_df.arg_t_vyst_r;
multiple_shot_miscalibration_df.arg_t = multiple_shot_miscalibration_df.arg_t_vyst;
multiple_shot_miscalibration_df.ramsey_miscalib = @. exp(-multiple_shot_miscalibration_df.arg_t_r * (1+multiple_shot_miscalibration_df.gamma2_miscalib_percent/100)/multiple_shot_miscalibration_df.t2) * sin(2*pi*multiple_shot_miscalibration_df.detune_ratio * multiple_shot_miscalibration_df.arg_t_r / multiple_shot_miscalibration_df.t2) / sqrt(multiple_shot_miscalibration_df.arg_t_r);
multiple_shot_miscalibration_df.improvement = @. multiple_shot_miscalibration_df.vyst / multiple_shot_miscalibration_df.ramsey_miscalib;
multiple_shot_miscalibration_df[(multiple_shot_miscalibration_df[:,:gamma1_miscalib_percent] .== 8) .& (multiple_shot_miscalibration_df[:,:gamma2_miscalib_percent] .== -10) .& (multiple_shot_miscalibration_df[:,:detune_ratio] .== 0.1) .& (multiple_shot_miscalibration_df[:,:t1us] .== 51),:improvement][1] = 0.5*(1.068677 + 1.070491)

# Decide which df to use
df = multiple_shot_miscalibration_df;
choice = "multiple";
df = single_shot_miscalibration_df;
choice = "single";

# Figure 6
# Plot change in max vy as a function of miscalibration in T1, T2
buff_df = filter([:t1us, :detune_ratio] => (x,y) -> x == 100 && y == 0.01, df);
for temp_df in groupby(buff_df, [:t1us, :detune_ratio])
    xs, ys, zs = create_heatmap_data(temp_df.gamma1_miscalib_percent, temp_df.gamma2_miscalib_percent, temp_df.improvement)
    graph = plot();
    max_diff = get_max_deviation_from_1(temp_df.improvement);
    plot!(graph, xs, ys, zs,
        title="Calibration sensitivity - SNR(Ts)/SNR(Tr), (T1/T2=$(temp_df.t1us[1]/temp_df.t2us[1]), Detuning/gamma2 = $((temp_df.detune_ratio[1])))",
        xlabel="Gamma1 miscalibration percent", ylabel="Gamma2 miscalibration percent",
        st=:heatmap, interpolate=true, c=:seismic, clims=(1-max_diff, 1+max_diff),
        grid=true, gridlinewidth=3, gridstyle=:solid,
        size=(1500,800), linewidth=linewidth,
            leftmargin=10mm, bottommargin=10mm);
    display(graph);
    savefig(graph, "$(choice) - calibration vs detuning (T1:T2=$(temp_df.t1us[1]/temp_df.t2us[1]), Detuning:gamma2 = $((temp_df.detune_ratio[1]))).svg");
end

## Plotting
# labels=["T1 - $(x.t1) us" for x in eachrow(unique(buff_df,:t1))];
# plot(x, y, title="text $(variable) us", ylims=(0,20000),
#    xlabel="X axis", ylabel="Y axis", labels=reshape([x*" (CM max)" for x in labels], 1, length(labels)),
#    linestyle=:dash, seriestype=:scatter, size=(1500,800), show=true);

## DF Manipulation

# Count unique values in column
# combine(groupby(df, :t1t2), nrow=>:count)

# Change ticks
# xticks=([e^x for x in range(1,10,steps=10), ["$(x)" for x in range(1,10,steps=10)]); # Logscale

# List each column
# show(names(df))

# Check element types
# eltype.(eachcol(df))

# Number of rows
# size(df)

# Delete at row
# delete!(df, []);

# Recast column
# df.t1 = parse.(Float64, df.t1);
#
# Filter df
# filter([:ramsey_vy_max,:cm_vy_end]=>(x,y)->x<y,df)
#
# Transform/create new column
# transform!(df, [:t1,:t2,:detuning,:vx] => ByRow(calculate_predicted_vy_end) => [:predicted_vz,:predicted_H,:predicted_vy_end,:valid]);
