local argcheck = require 'argcheck'

local tnt   = require 'torchnet'
local utils = require 'torchnet-vision.datasets.utils'

local m2caiworkflow = {}

-- m2caiworkflow.__preprocess = argcheck{
--    {name='dirname', type='string', default='data/raw/m2caiworkflow'},
--    call =
--       function(dirname)
--          --os.execute('unzip '..dirname..'/roof_images.zip -d '..dirname)
--          local pathvideo      = paths.concat(dirname, 'video')
--          local pathvideotest  = paths.concat(dirname, 'videoTest')
--          local pathimages     = paths.concat(dirname, 'images')
--          local pathimagestest = paths.concat(dirname, 'imagesTest')
--          os.execute([[for video in `ls videos`; do
--                         ffmpeg -i ]]..pathvideo.. [[/$video -vf fps=1 ]]
--                                     ..pathimages..[[/${video%.*}-%4d.jpg
--                       done]])
--          os.execute([[for video in `ls videosTest`; do
--                         ffmpeg -i ]]..pathvideotest.. [[/$video -vf fps=1 ]]
--                                     ..pathsimagesTest..[[/${video%.*}-%4d.jpg
--                       done]])
                  
-- }

m2caiworkflow.load  = argcheck{
   {name='dirname', type='string', default='data/raw/m2caiworkflow'}
   {name='part', type='string', default='1'},
   call =
   function(dirname, part)
      if not paths.dirp(dirname..'/images'):
         self.__preprocess(dirname)
      end
      local pathimages   = paths.concat(dirname,'images')
      local pathtraintxt = paths.concat(dirname,'dataset2/trainset_'..part..'.txt')
      local pathvaltxt   = paths.concat(dirname,'dataset2/valset_'..part..'.txt')
      local trainset = tnt.ListDataset{
          filename = pathtraintxt,
          path = pathimages,
          load = function(line)
             local sample = {line=line}
             return sample
          end
      }
      local valset = tnt.ListDataset{
          filename = pathvaltxt,
          path = pathimages,
          load = function(line)
             local sample = {line=line}
             return sample
          end
      }
      local classes = {"TrocarPlacement", "Preparation",
         "CalotTriangleDissection", "ClippingCutting",
         "GallbladderDissection", "GallbladderPackaging",
         "CleaningCoagulation", "GallbladderRetraction"}
      local class2target = {}
      for k,v in pairs(classes) do class2target[v] = k end
      return trainset, valset, classes, class2target
   end
}

m2caiworkflow.loadTrainset = function()
   local pathimages   = paths.concat(dirname,'images')
   local pathtraintxt = paths.concat(dirname,'dataset2/trainset_'..part..'.txt')
   local trainset = tnt.ListDataset{
       filename = pathtraintxt,
       path = pathimages,
       load = function(line)
          local sample = {line=line}
          return sample
       end
   }
   local classes = {"TrocarPlacement", "Preparation",
         "CalotTriangleDissection", "ClippingCutting",
         "GallbladderDissection", "GallbladderPackaging",
         "CleaningCoagulation", "GallbladderRetraction"}
      local class2target = {}
      for k,v in pairs(classes) do class2target[v] = k end
   return trainset, classes, class2target
end

m2caiworkflow.loadTestset = function()
   local pathimages   = paths.concat(dirname,'imagesTest')
   local pathimages   = '/local/robert/m2cai/workflow/imagesTest'
   local pathtraintxt = paths.concat(dirname,'dataset2/testset.txt')
   local testset = tnt.ListDataset{
       filename = pathtesttxt,
       path = pathimages,
       load = function(line)
          local sample = {line=line}
          return sample
       end
   }
   return testset
end

return m2caiworkflow
