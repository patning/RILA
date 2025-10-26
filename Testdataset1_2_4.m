rng(1); % Sets the seed to 1

%% EVALUATION SCRIPT - DATA GEN BY 2,4 Fully Quantum HQMM

SEQ_LENGTH=100;
NUM_TR_DATA=30;
NUM_TE_DATA=5;

HIDDEN = 2;
OBS = 4;
burn_in = 0;


%% HQMM Synthetic Data -- 2 states, 4 outputs
K_true_24quant = cell(4,1);
K_true_24quant{1} = [1/sqrt(2), 0; 0,0];
K_true_24quant{2} = [0, 0; 0,1/sqrt(2)];
K_true_24quant{3} = [1/(2*sqrt(2)), 1/(2*sqrt(2)); 1/(2*sqrt(2)),1/(2*sqrt(2))];
K_true_24quant{4} = [1/(2*sqrt(2)), -1/(2*sqrt(2)); -1/(2*sqrt(2)),1/(2*sqrt(2))];
rho_true_24quant = [1,0;0,0];

train_data_24quant = generateObs(K_true_24quant, rho_true_24quant, NUM_TR_DATA, SEQ_LENGTH);

% Corrupt exactly 10 sequences
num_corrupt = 0;
corrupt_indices = randperm(NUM_TR_DATA, num_corrupt);
adversarial_seq = ones(1, SEQ_LENGTH) * 4;  % Adversarial sequence: all 4s (adjust if needed)
disp(sort(corrupt_indices));
train_data_corrupt = train_data_24quant;
for ii = 1:num_corrupt
    i = corrupt_indices(ii);
    train_data_corrupt(i, :) = adversarial_seq;
end

test_data_24quant = generateObs(K_true_24quant, rho_true_24quant, NUM_TE_DATA, SEQ_LENGTH);
[m,s] = llstats(train_data_corrupt, K_true_24quant, rho_true_24quant, burn_in);
[m2,s2] = llstats(test_data_24quant, K_true_24quant, rho_true_24quant, burn_in);
fprintf('\nTrue HQMM Values: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
%True HQMM Values: -104.278519 +/- 3.417438, Val LL: -105.358371 +/- 5.757713

save('/Users/patricianing/Desktop/projects/Learning Hidden Quantum Markov Models/Matlab_codes/Data/HQMM_2_4_1.mat','K_true_24quant', 'rho_true_24quant','train_data_corrupt', 'test_data_24quant');  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1); % Sets the seed to 1

burn_in = 0;
[~,TR,EM] = generateHMMParams(HIDDEN,OBS);
[ESTTR, ESTEM] = hmmtrain(train_data_corrupt(:,burn_in+1:end),TR',EM','verbose',true,'maxiterations',10000,'tolerance',7e-6);
[m,s] = llstatsclass(train_data_corrupt, ESTTR', ESTEM', [1;0], burn_in);
[m2,s2] = llstatsclass(test_data_24quant, ESTTR', ESTEM', [1;0], burn_in);
fprintf('\nBest Learned HMM Train LL: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
%Best Learned HMM Train LL: -138.389321 +/- 0.886904, Val LL: -138.656472 +/- 0.597659

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train HQMM from scratch
load('/Users/patricianing/Desktop/projects/Learning Hidden Quantum Markov Models/Matlab_codes/Data/HQMM_2_4_1.mat');  

rng(1); % Sets the seed to 1

HIDDEN = 2;
OBS = 4;

burn_in = 0;
batch_size = 5;
iter = 6;
num_batches = 4;


Kguess = generateHQMM(HIDDEN,OBS,1);
rho_guess= zeros(HIDDEN,HIDDEN);
rho_guess(1,1)=1;

k = 10; 
tStart = tic;
K_24quant = RlearnHQMM(HIDDEN, @llstats, train_data_corrupt, test_data_24quant, Kguess, rho_guess, burn_in, batch_size, iter, num_batches, k, 0, 'patternsearch');
[m,s] = llstats(train_data_corrupt, K_24quant, rho_guess, burn_in);
[m2,s2] = llstats(test_data_24quant, K_24quant, rho_guess, burn_in);
fprintf('\nBest Learned HQMM Train LL: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
elapsedTime = toc(tStart);  % returns the elapsed time in seconds
disp(elapsedTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('/Users/patricianing/Desktop/projects/Learning Hidden Quantum Markov Models/Matlab_codes/Data/HQMM_2_4_1.mat');  

rng(1); % Sets the seed to 1

HIDDEN = 2;
OBS = 4;

burn_in = 0;
batch_size = 5;
iter = 6;
num_batches = 4;


Kguess = generateHQMM(HIDDEN,OBS,1);
rho_guess= zeros(HIDDEN,HIDDEN);
rho_guess(1,1)=1;

k = 10; 
tStart = tic;
K_24quant = RlearnHQMM(HIDDEN, @llstats, train_data_corrupt, test_data_24quant, Kguess, rho_guess, burn_in, batch_size, iter, num_batches, k, 0, 'fmincon');
[m,s] = llstats(train_data_corrupt, K_24quant, rho_guess, burn_in);
[m2,s2] = llstats(test_data_24quant, K_24quant, rho_guess, burn_in);
fprintf('\nBest Learned HQMM Train LL: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
elapsedTime = toc(tStart);  % returns the elapsed time in seconds
disp(elapsedTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train HQMM from scratch
load('/Users/patricianing/Desktop/projects/Learning Hidden Quantum Markov Models/Matlab_codes/Data/HQMM_2_4_1.mat');  

rng(1); % Sets the seed to 1

HIDDEN = 2;
OBS = 4;

burn_in = 0;
batch_size = 5;
iter = 6;
num_batches = 8;


Kguess = generateHQMM(HIDDEN,OBS,1);
rho_guess= zeros(HIDDEN,HIDDEN);
rho_guess(1,1)=1;

k = 10; 
tStart = tic;
K_24quant = RlearnHQMM(HIDDEN, @llstats, train_data_corrupt, test_data_24quant, Kguess, rho_guess, burn_in, batch_size, iter, num_batches, k, 0, 'fmincon');
[m,s] = llstats(train_data_corrupt, K_24quant, rho_guess, burn_in);
[m2,s2] = llstats(test_data_24quant, K_24quant, rho_guess, burn_in);
fprintf('\nBest Learned HQMM Train LL: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
elapsedTime = toc(tStart);  % returns the elapsed time in seconds
disp(elapsedTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('/Users/patricianing/Desktop/projects/Learning Hidden Quantum Markov Models/Matlab_codes/Data/HQMM_2_4_1.mat');  

rng(1); % Sets the seed to 1

HIDDEN = 2;
OBS = 4;

burn_in = 0;
batch_size = 5;
iter = 6;
num_batches = 4;


Kguess = generateHQMM(HIDDEN,OBS,1);
rho_guess= zeros(HIDDEN,HIDDEN);
rho_guess(1,1)=1;

tStart = tic;
K_24quant = learnHQMM(HIDDEN, OBS, train_data_corrupt, test_data_24quant, Kguess, rho_guess, burn_in, batch_size, iter, num_batches);
[m,s] = llstats(train_data_corrupt, K_24quant, rho_guess, burn_in);
[m2,s2] = llstats(test_data_24quant, K_24quant, rho_guess, burn_in);
fprintf('\nBest Learned HQMM Train LL: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
elapsedTime = toc(tStart);  % returns the elapsed time in seconds
disp(elapsedTime);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('/Users/patricianing/Desktop/projects/Learning Hidden Quantum Markov Models/Matlab_codes/Data/HQMM_2_4_1.mat');  

rng(1); % Sets the seed to 1

HIDDEN = 2;
OBS = 4;

burn_in = 0;
batch_size = 5;
iter = 6;
num_batches = 8;


Kguess = generateHQMM(HIDDEN,OBS,1);
rho_guess= zeros(HIDDEN,HIDDEN);
rho_guess(1,1)=1;

tStart = tic;
K_24quant = learnHQMM(HIDDEN, OBS, train_data_corrupt, test_data_24quant, Kguess, rho_guess, burn_in, batch_size, iter, num_batches);
[m,s] = llstats(train_data_corrupt, K_24quant, rho_guess, burn_in);
[m2,s2] = llstats(test_data_24quant, K_24quant, rho_guess, burn_in);
fprintf('\nBest Learned HQMM Train LL: %f +/- %f, Val LL: %f +/- %f\n', m, s, m2, s2);
elapsedTime = toc(tStart);  % returns the elapsed time in seconds
disp(elapsedTime);



