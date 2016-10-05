require 'paths'

local function findPathlogs(pathdir)
   local pathlogs = {}
   for k,path in ipairs({pathdir}) do
      local dirs = dir.getdirectories(path);
      for k,dirpath in ipairs(dirs) do
         local exp = paths.basename(dirpath)
         table.insert(pathlogs, exp)
      end
   end
   table.sort(pathlogs, function(a, b) return a < b end)
   return pathlogs
end

local function loadLogs(pathdir)
   local logs = {}
   local pathlogs = findPathlogs(pathdir)
   --print(pathlogs)
   for i,pathlog in pairs(pathlogs) do
      local pathconfig = pathdir..'/'..pathlog..'/config.t7'
      local pathbestepoch = pathdir..'/'..pathlog..'/bestepoch.t7'
      local config = torch.load(pathconfig)
      --print(config)
      print(pathbestepoch)
      if paths.filep(pathbestepoch) then
         local bestepoch = torch.load(pathbestepoch)
         print(bestepoch)
         if bestepoch.clerrtop1 then
            bestepoch.acctop1 = 100 - bestepoch.clerrtop1
         end
         table.insert(logs, {config=config, bestepoch=bestepoch})
         --print(logs)
      end
   end
   --print(logs)
   table.sort(logs, function(a, b)
      return a.bestepoch.acctop1 < b.bestepoch.acctop1
   end)
   return logs
end

local cmd = torch.CmdLine()
cmd:option('-pathdir', 'logs/m2caiworkflow/resnet', '')
-- cmd:option('-dispepoch', false, '')
local config = cmd:parse(arg)

local logs = loadLogs(config.pathdir)
print(logs)
print('')
print('Tableau de résultats pour '..config.pathdir)
print('')
print('| lr | lrd | bsize | epoch | acctop1 |')
print('|----|-----|-------|-------|---------|')
for i,log in pairs(logs) do
   print('| '..log.config.lr
      ..' | '..log.config.lrd
      ..' | '..log.config.bsize
      ..' | '..log.bestepoch.epoch
      ..' | '..log.bestepoch.acctop1
      ..' |')
end
print('')
print('Dans le même ordre qu\'au dessus :')
print('')
for i,log in pairs(logs) do
   print('- '..config.pathdir..'/'..log.config.date..'/net.t7')
end
