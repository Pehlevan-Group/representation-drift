%==========================================================
% MATLAB job submission script: parallel_batch.m
%==========================================================
c = parcluster('odyssey');
c.AdditionalProperties.QueueName = 'shared';
c.AdditionalProperties.WallTime = '05:00:00';
c.AdditionalProperties.MemUsage = '5000';
j = c.batch(@PSP_D_Dependency, 1, {}, 'pool', 100);
exit;