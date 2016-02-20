function generate_json(input_mat_file, output_file)
% This code is to generate the parameters for population models

out_struct = struct();
% load the appliance time series
po = load(input_mat_file);
applianceNames = fieldnames(po)';
numAppliances = length(applianceNames);
for i=1:numAppliances
    appliance_name = applianceNames{i};
    dataAppliance = po.(appliance_name);
    dataAppliance(find(isnan(dataAppliance))) = 0;

    K = 2;

    % We use the mean to for each cluster to generate the HMM states; and the
    % threshold was 5; you can define this yourself.
    method = 'mean';
    smallValue = 5;

    addpath(genpath('./generatePopulationModels'));
    % Given the kettle data for many days, this function generate the HMM state
    % means and transfer the data to state time series.
    [Appliance_state, Appliance_stateMean, Appliance_data] = clusterStateAppliance(dataAppliance, K, method, smallValue);

    % This function is to generate the sumarry statistics
    % t_activeTime: the total active time for each day
    % nosOfActCycle: the total number of cycles for each day
    % e_totalConsumption: the total energy used for each day
    [t_activeTime, t_OnOff, t_OffOn, nosOfActCycle, t_duration, ...
              e_totalConsumption, e_ave_totalConsumption_cyc, e_ave_totalConsumption_day, ...
              p_ave_power_cyc, p_ave_power_day]=extract_feature(Appliance_data,Appliance_state,Appliance_stateMean);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This is to generate the parameters for population models
    maxNosOfCycle = 15;
    if strcmp('fridge', appliance_name)
        minNosOfCycle = 5;
    else
        minNosOfCycle = 0;
    end
        
    

    minCycle = max(0, minNosOfCycle);
    maxCycle = min(maxNosOfCycle, max(nosOfActCycle));
    nos_of_cycle = length(nosOfActCycle);
    numberOfCyclesProb = zeros(maxCycle-minNosOfCycle+1,1);
    numberOfCyclesDuration = zeros(maxCycle-minNosOfCycle+1,1);
    numberOfCyclesEnergy = zeros(maxCycle-minNosOfCycle+1,1);
    j = 0;
    for i=minCycle:maxCycle
        j = j + 1;
        numberOfCyclesProb(j)=length(find(nosOfActCycle==i));
        numberOfCyclesDuration(j)=mean(t_activeTime(find(nosOfActCycle==i)));
        numberOfCyclesEnergy(j)=mean(e_totalConsumption(find(nosOfActCycle==i)));
    end
    numberOfCyclesProb = numberOfCyclesProb/sum(numberOfCyclesProb);

    % set NaN to zero
    numberOfCyclesProb(find(isnan(numberOfCyclesProb))) = 0;
    numberOfCyclesDuration(find(isnan(numberOfCyclesDuration))) = 0;
    numberOfCyclesEnergy(find(isnan(numberOfCyclesEnergy))) = 0;

    % the number of cycles
    numberOfCycles = minCycle:maxCycle

    induced_density_of_duration = [mean(numberOfCyclesDuration), std(numberOfCyclesDuration)];
    induced_density_of_sac = [mean(numberOfCyclesEnergy), std(numberOfCyclesEnergy)];
    numberOfCycles = numberOfCycles;
    numberOfCyclesDuration = reshape(numberOfCyclesDuration, [length(numberOfCyclesDuration),1]);
    numberOfCyclesEnergy = reshape(numberOfCyclesEnergy, [length(numberOfCyclesEnergy),1]);
    numberOfCyclesProb = reshape(numberOfCyclesProb, [length(numberOfCyclesProb),1]);

    % Now get the HMM parameters
    addpath(genpath('./generate_paramsOfHMMs'));

    K = 3;

    % find the HMM parameters for upright freezer 1
    [P_S, Pi_S, P_S_freq, Pi_S_freq, C, Mode_X, XID_matrix] = findProbAppliance(dataAppliance, K);

    means = Mode_X;
    startprob=Pi_S;
    transprob=P_S;
    numberOfStates=length(Mode_X);

    addpath(genpath('./jsonlab-1.2'));

    num_cycles_struct = struct('numberOfCycles', numberOfCycles,...
        'numberOfCyclesDuration', numberOfCyclesDuration',...
        'numberOfCyclesEnergy', numberOfCyclesEnergy',...
        'numberOfCyclesProb', numberOfCyclesProb');

    json_struct=struct('induced_density_of_duration', induced_density_of_duration,...
    'induced_density_of_sac', induced_density_of_sac,...
    'numberOfCyclesStats', num_cycles_struct,...
    'means',means, 'startprob', startprob,...
    'transprob',transprob, 'numberOfStates', numberOfStates)
    
    %%NOW SAVING THE DATA OF APPLIANCE
    out_struct.(appliance_name) = json_struct;
end

%% Create JSON
json_string = savejson('', out_struct);
json_string = strrep(json_string, 'induced_density_of_duration', 'induced density of duration');
json_string = strrep(json_string, 'induced_density_of_sac', 'induced density of sac');

fprintf(fopen(output_file,'w'),'%s',json_string);

