local argcheck = require 'argcheck'

local tnt   = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'

local m2caiworkflow = {}

m2caiworkflow._classes = {"TrocarPlacement", "Preparation",
   "CalotTriangleDissection", "ClippingCutting",
   "GallbladderDissection", "GallbladderPackaging",
   "CleaningCoagulation", "GallbladderRetraction"}

local class2target = {}
for k,v in pairs(m2caiworkflow._classes) do class2target[v] = k end
m2caiworkflow._class2target = class2target

m2caiworkflow._load = function(dirimg, pathtxt)
   local dataset = tnt.ListDataset{
       filename = pathtxt,
       path = dirimg,
       load = function(line)
          local sample = {line=line}
          return sample
       end
   }
   return dataset
end

m2caiworkflow.load  = argcheck{
   {name='dirimg',       type='string', default='data/interim/images'},
   {name='pathtraintxt', type='string', default='data/interim/trainset.txt'},
   {name='pathvaltxt',   type='string', default='data/interim/valset.txt'},
   call =
   function(dirimg, pathtraintxt, pathvaltxt)
      local trainset = m2caiworkflow._load(dirimg, pathtraintxt)
      local valset   = m2caiworkflow._load(dirimg, pathvaltxt)
      return trainset, valset, m2caiworkflow._classes, m2caiworkflow._class2target
   end
}

m2caiworkflow.loadFullTrainset = argcheck{
   {name='dirimg',  type='string', default='data/interim/images'},
   {name='pathtxt', type='string', default='data/interim/fulltrainset.txt'},
   call =
   function(dirimg, pathtxt)
      local trainset = m2caiworkflow._load(dirimg, pathtxt)
      return trainset, m2caiworkflow._classes, m2caiworkflow._class2target
   end
}

m2caiworkflow.loadTestset = argcheck{
   {name='dirimg',  type='string', default='data/interim/imagesTest'},
   {name='pathtxt', type='string', default='data/interim/testset.txt'},
   call =
   function(dirimg, pathtxt)
      local testset = m2caiworkflow._load(dirimg, pathtxt)
      return testset
   end
}

return m2caiworkflow
