clear all
clc


X = importdata('x.txt');
d = importdata('d.txt');
listing = dir('outputs');
folders_temp = {listing.name};
for i = 3:length(folders_temp)
    folders{i-2} = folders_temp{i};
    fold_num(i-2) =  str2num(folders_temp{i});
end
[fold_num,i_ordered]= sort(fold_num);
temp = folders;
for i = 1:length(folders)
  folders{i} = temp{i_ordered(i)};
end




v = VideoWriter('video3');
v.FrameRate = 10;

open(v);
figure('WindowState', 'maximized')
for i = 1:length(folders)
    BS = importdata(strcat('outputs\',folders{i},'\break_switch.txt'));
    data = importdata(strcat('outputs\',folders{i},'\zeta.txt'));
    plot(X,data)
    hold on
    plot(X,d)
    plot(X,BS)
    %hold on
    %plot(X,-d)
    plot(X,zeros(length(X),1),'k--')
 
    ylim([d(1) 4])
   
    hold off
   
    
    
   
    title(strcat('t  =  ',num2str(round(str2num(folders{i}),6))))
  
    frame =getframe(gcf);
    writeVideo(v,frame);
end
close(v);