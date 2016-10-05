local utils = {}

utils.iterOverDataset = function(dataset, maxiter, hook)
   local i = 1
   for sample in dataset:shuffle():iterator()() do
      if hook.onSample then hook.onSample(sample) end
      xlua.progress(i, maxiter)
      i = i + 1
      if i >= maxiter then break end
   end
   if hook.onEnd then hook.onEnd() end
end

utils.processMeanStd = function(dataset, pc, mean, std)
   require 'xlua'
   local maxiter = torch.round(dataset:size() * pc)
   print('Process mean')
   utils.iterOverDataset(dataset, maxiter, {
      onSample = function(sample) mean:add(sample.input) end,
      onEnd = function() mean:div(maxiter) end
   })
   print('')
   print('Process std')
   utils.iterOverDataset(dataset, maxiter, {
      onSample = function(sample) std:add((sample.input - mean):pow(2)) end,
      onEnd = function() std:div(maxiter) end
   })
   print('')
   return mean, std
end

return utils
