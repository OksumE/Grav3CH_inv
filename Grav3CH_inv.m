function Grav3CH_inv
clc;clear all;clear global;
%Code by 
%Oksum. E. : eroksum@gmail.com / erdincoksum@sdu.edu.tr 
% 2020
%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Brief Descriptions %%%%%%%%%
%%%%%%%%% INVERSION/Forward Calculation 
     % 1- Import gridded Gravity/Depth data [2-D grid (*.grd]
     %    by Load Data menu at main gui
     % 2- Set Density parameters r0 and lambda in Settings 1 panel 
     %    Options: constant >> the value can be constant for all prism models
     %             Variable in x,y location >> grid required to be in same size of input gravity grid... 
     % 4- [IF OPERATION IS INVERSION]
     %    Set Iteration stop mode/criterion and maximum number of iteration to 
     %    the related cells in Settings 2 panel 
     % 3- Press [Forward Calc] or the [Start Inversion] pushbutton
     %    If input is Depth model then [Forward Calc] is enabled on automatically
     %    otherwise If input is Gravity data then [Start Inversion] is enabled on
     %
     % View Outputs : Basement depth, Inverted anomaly, Anomaly Difference, 
     %                RMS plot, Max.error plot. 
     %                (interactively selectable by a mouse clicks on relevant menu items
     % View Options :  Cross-Section (for depth,observed and calculated anomalies)
     % Export Outputs: Image file (*.png, 300 dpi resolution),
     %                 Data file (maps-> *.grd, graphics-> *.dat)
     %                 All outputs are auto-stored into specified directory created by the
     %                 user. The code adds extension to the output
     %                 *.-grav3CH_gobs.grd >>
     %                 *.-grav3CH_gcalc.grd >>
     %                 *.-grav3CH_gdiff.grd >>
     %                 *.-grav3CH_zcalc.grd >>
     %                 *.-grav3CH_rmserr.dat >>
     %                 *.-grav3CH_project.mat >>
     %                 Project files can be loaded any time for re-view of interpretation
%%%%%%%%%%%%%%% Tasks of some main Functions
%%% datainput > NEW 2-D GRID and initialize/memorize parameters
                % uses lodgrd6txt,lodgrd7bin for reading grid (*.grd) format
%%% startfunc > retrievs inputs, performs inversion/forward
             %  memorize outputs, creates output menu windows
%%% maininv_CH_CH > performs inversion sheme
%%% FW_CH    > performs forward calculation
%%% freqaxTG > calculates wavenumbers k, kx,ky and variables standing outside 
             % the loop of forward  

%%% please see Manual pdf file for the tasks of all functions used
%%% in code
drawnow;delete(findobj('Type','figure','name','MAP'));
create_MainGui
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT DATA functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function datainput(~,~,modeDT,ax)
if modeDT==0;openproj(ax);return;end
%%%%%% popup input window
drawnow
[filename, pathname] = uigetfile('*.grd', 'Import Golden Software Binary/Text grid (*.grd)');
sourcfil=[pathname filename];
if ischar(sourcfil)
[~,filenam] = fileparts(sourcfil);    
drawnow;delete(findobj('Type','figure','name','MAP'));
resetmainwindow
fidc=fopen(sourcfil);header= fread(fidc,4,'*char' )';fclose(fidc);
c1=strcmp(header,'DSAA');
c2=strcmp(header,'DSRB');
sumc=sum([c1 c2]);
if sumc>0
switch c1
    case 1
[matrix,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy]=lodgrd6txt(sourcfil); %format surfer6 text
    case 0
[matrix,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy]=lodgrd7bin(sourcfil); %surfer7 binary       
end
%%%%%%%%%%% print grid info
if modeDT==1 || modeDT==-1
mapinfo(matrix,nx,ny,xmin,xmax,ymin,ymax,dx,dy,filename)
end
%%%%%%%%%%%%% check matrix size even or odd numbered..if odd 
%%%%%%%%%%%%% initialise to even size by cutting one column or one row at
%%%%%%%%%%%%% east or north end 
if mod(nx,2)==1;nx=nx-1;matrix(:,end)=[];x(:,end)=[];y(:,end)=[];xmax=max(max(x));end
if mod(ny,2)==1;ny=ny-1;matrix(end,:)=[];y(end,:)=[];x(end,:)=[];ymax=max(max(y));end
if dx~=dy; error_message(2);return;end
if any(isnan(matrix(:)));error_message(4);return;end
else
error_message(3);return;    
end
%%%%%%%%%%%%%%%%%%%%%%%%%
cla(ax,'reset');axis off;
%%% case input is Gravity or Depth grid
if modeDT==1 || modeDT==-1
data=matrix;
if modeDT==1;data=(abs(data));end % depth is always positive
save('tempinv.mat','data','x','y','nx','ny','xmin','xmax','ymin','ymax','dx','dy',...
'filename','modeDT','-append');
switch modeDT
    case 1
    set(findobj(gcf,'Tag','frwbut'),'enable','on')
    set(findobj(gcf,'Tag','invbut'),'enable','off')
    case -1
    set(findobj(gcf,'Tag','frwbut'),'enable','off')
    set(findobj(gcf,'Tag','invbut'),'enable','on')
end
showmap(0,0,0)
end
%%% case input density model grid
if modeDT==11
r02d=matrix;
filenam11=filenam;
save('tempinv.mat','r02d','filenam11','-append')
set(findobj(gcf,'Tag','pushimp11'),'string',filenam)
end
%%% case input decay (lambda) model grid
if modeDT==22
lambda2d=matrix;
filenam22=filenam;
save('tempinv.mat','lambda2d','filenam22','-append')
set(findobj(gcf,'Tag','pushimp22'),'string',filenam)
end
end
end

function openproj(ax) %%% load a project file, created by grav3CH_inv
[filename, pathname] = uigetfile('*.mat', 'Open Poject file');
kk=[pathname filename];
if ischar(kk)
%%%%%%close figures and clean axes
drawnow;delete(findobj('Type','figure','name','MAP'));
cla(ax,'reset');axis off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
copyfile(kk,'tempinv.mat');
load('tempinv.mat');
%%%%%%%%%%%% retriew settings and set control window objects;
mapinfo(data,nx,ny,xmin,xmax,ymin,ymax,dx,dy,filename)
unqr0=unique(r0);unqlambda=unique(lambda);
%%%%%%%%%%%%%%%%%%
if numel(unqr0)==1
set(findobj(gcf,'Tag','edr0'),'string',num2str(unqr0))
set(findobj(gcf,'Tag','listb2'),'value',1,'enable','off')
set(findobj(gcf,'Tag','pushshwmp11'),'enable','off')
else
set(findobj(gcf,'Tag','edr0'),'string','')    
set(findobj(gcf,'Tag','listb2'),'value',2,'enable','off')
set(findobj(gcf,'Tag','pushshwmp11'),'enable','on')
set(findobj(gcf,'Tag','pushimp11'),'string',filenam11)
end
if numel(unqlambda)==1
set(findobj(gcf,'Tag','edlambda'),'string',num2str(unqlambda))
set(findobj(gcf,'Tag','listb3'),'value',1,'enable','off')
set(findobj(gcf,'Tag','pushshwmp22'),'enable','off')
else
set(findobj(gcf,'Tag','edlambda'),'string','')    
set(findobj(gcf,'Tag','listb3'),'value',2,'enable','off')
set(findobj(gcf,'Tag','pushshwmp22'),'enable','on')
set(findobj(gcf,'Tag','pushimp22'),'string',filenam22)
end
set(findobj(gcf,'Tag','listb4'),'value',modetermin,'enable','off')
set(findobj(gcf,'Tag','edcrit'),'string',num2str(criterio))
set(findobj(gcf,'Tag','edmxiter'),'string',num2str(maxiter))
set(findobj(gcf,'Tag','edign'),'string',num2str(wedge))
set(findobj(gcf,'Tag','frwbut'),'enable','off')
set(findobj(gcf,'Tag','invbut'),'enable','off')
set(findobj(gcf,'Tag','pushimp11'),'enable','off')
set(findobj(gcf,'Tag','pushimp22'),'enable','off')
drawnow;create_actwindow;
mapper(x,y,-zcalc,'Km','Calculated Depth',wedge,'cntrf')
outmenu_inversion(x,y,data,zcalc,gcalc,wedge,rmstor,erstor)
end
end

function load_dens_measure(~,~,menuobj3)
[filename, pathname] = uigetfile({'*.dat';'*.txt'}, 'Import data');
k=[pathname filename];
if ischar(k)
set(menuobj3,'label',[ ' File: ' filename])    
drawnow;cla;drawnow    
depdensm=load(k);
dep=abs(depdensm(:,1));
contrast=abs(depdensm(:,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculate exp fit parametes; r0 and lambda cofficients
f=fit(-dep,-contrast,'exp1');
r0=f.a; lambda=f.b;
ci=confint(f);
fitcur=r0.*exp(lambda.*-dep);
fitcurC1=ci(1,1).*exp(ci(1,2).*-dep);
fitcurC2=ci(2,1).*exp(ci(2,2).*-dep);
plot(-contrast,-dep,'k--','linewidth',2);hold on
plot(fitcur,-dep,'-r','linewidth',2);
plot(fitcurC1,-dep,'g-','linewidth',2);
plot(fitcurC2,-dep,'g-','linewidth',2);
str1=['r0= ' num2str(r0) '  lambda= ' num2str(lambda)];
str2=['r0= ' num2str(ci(1,1)) '  lambda= ' num2str(ci(1,2))];
str3=['r0= ' num2str(ci(2,1)) '  lambda= ' num2str(ci(2,2))];
hl=legend('measured density contrast-depth data',str1,str2,str3);
set(hl,'location','SouthEast')
hold off
set(gca, 'XDir','reverse','xaxisLocation','top','FontSize',10,'FontWeight','bold');
xlabel('Density contrast (gr/cm3)');
ylabel('Depth (Km)');
title(['Exponential curve fitting to measured density contrast-depth data'...
    ' (with 95% confidence bounds) >> ro= r0*exp(-lambda*z)'])
grid on
% contrvec=-contrast;
% depvec=-dep;
% fitcurvec=fitcur;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% File reader/writer functions
%%%%%read Surfer 6 text grid(*.grd)
function [T,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy]=lodgrd6txt(k)
surfergrd=fopen(k,'r'); % Open *.grid file
dsaa=fgetl(surfergrd);  % Header
% Get the map dimension [NX: East NY: North];
datasize=str2num(fgetl(surfergrd)); nx=datasize(1); ny=datasize(2);
% Map limits: xmin, xmax, ymin ymax
xcoor=str2num(fgetl(surfergrd)); xmin=xcoor(1); xmax=xcoor(2);
ycoor=str2num(fgetl(surfergrd)); ymin=ycoor(1); ymax=ycoor(2);
% check intervals in x and y direction 
dx=(xmax-xmin)/(nx-1);dx=abs(dx);
dy=(ymax-ymin)/(ny-1);dy=abs(dy);
% data limits
anom=str2num(fgetl(surfergrd)); t0min=anom(1); t0max=anom(2);
% data matrix 
[T,numb] = fscanf(surfergrd, '%f', [nx,ny]);
T=T'; % Traspose matrix
fclose(surfergrd);
% map coordinate matrix
[x,y]=meshgrid(xmin:dx:xmax,ymin:dy:ymax);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%read Surfer 7 Binary grid
function [T,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy] = lodgrd7bin(filename)
fid= fopen(filename);
fread(fid,4,'*char' )';
fread(fid,1,'uint32');fread(fid,1,'uint32');
fread(fid,4,'*char' )';fread(fid,1,'uint32');
ny= fread(fid,1,'uint32'); nx= fread(fid,1,'uint32');
xmin= fread(fid,1,'double'); ymin= fread(fid,1,'double');
dx= fread(fid,1,'double'); dy= fread(fid,1,'double');
fread(fid,1,'double');fread(fid,1,'double');
fread(fid,1,'double');
parm= fread(fid,1,'double');
fread(fid,4,'*char' )';
nn= fread(fid,1,'uint32');
if ny*nx ~= nn/8 ; error('error') ;end
T= nan(nx,ny);
T(1:end) = fread(fid,numel(T),'double');
T=T';
fclose(fid);
T(T==parm) = nan;
xv = xmin + (0:nx-1)*dx;
yv = ymin + (0:ny-1)*dy;
[x,y]=meshgrid(xv,yv);
xmax=xv(end);
ymax=yv(end);
end
%%%%%%%%%%% Function for output of a GRID
function grdout(matrix,xmin,xmax,ymin,ymax,namefile)
%Get grid dimensions
aux=size(matrix);
nx=aux(2);ny=aux(1);
grdfile=fopen(namefile,'w');                % Open file
fprintf(grdfile,'%c','DSAA');               % Header code
fprintf(grdfile,'\n %i %i',[nx ny]);        % Grid size
fprintf(grdfile,'\n %f %f',[xmin xmax]);    % X limits
fprintf(grdfile,'\n %f %f',[ymin ymax]);    % Y limits
fprintf(grdfile,'\n %f %f',[min(min(matrix)) max(max(matrix))]); % Z limits
fprintf(grdfile,'\n');
for jj=1:ny                                 % Write matrix
    for ii=1:nx
        fprintf(grdfile,'%g %c',matrix(jj,ii),' ');
    end
    fprintf(grdfile,'\n');
end
fclose(grdfile);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Calculation Functions
function startfunc(~,~,modeDT,lstb2,lstb3,lstb4,lstb5,edr0,edlambda,edcrit,edmxiter,edign)
drawnow;delete(findobj('Type','figure','name','MAP'));
drawnow;set(lstb5,'value',2,'enable','on');
%%% retrievs inputs, settings
%%% call function inversion or forward
%%% memorize outputs
val2=get(lstb2,'value');val3=get(lstb3,'value');val4=get(lstb4,'value');
criterio=str2double(get(edcrit,'string'));criterio=abs(criterio);
maxiter=str2double(get(edmxiter,'string'));maxiter=abs(maxiter);
wedge=str2double(get(edign,'string'));wedge=abs(wedge);
modetermin=val4;
load('tempinv.mat','x','y','data','nx','ny','dx','dy')
%%%% initialize surface density contrast model
if val2==1;
    r0=str2double(get(edr0,'string'));r0=-abs(r0);r0=r0*ones(ny,nx);
else
    if isempty(who('-file', 'tempinv.mat', 'r02d'));error_message(6);return;end
    load('tempinv.mat','r02d');
    if isequal(size(x),size(r02d))==0;error_message(5);return;end
    r0=r02d;
end
%%%% initialize lambda decay rate model
if val3==1;
    lambda=str2double(get(edlambda,'string'));lambda=abs(lambda);
    if lambda==0;lambda=0.001;end;lambda=lambda*ones(ny,nx);
else
    if isempty(who('-file', 'tempinv.mat', 'lambda2d'));error_message(6);return;end
    load('tempinv.mat','lambda2d');
    if isequal(size(x),size(lambda2d))==0;error_message(5);return;end
    lambda=lambda2d;
end
%%%%%%%%%%%%%%%
%memorize settings
save('tempinv.mat','r0','lambda','criterio','maxiter','modetermin','wedge','-append')
%%%%%%%%%%%% PERFORM INVERSION OR FORWARD ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if modeDT==1; %%% FORWARD MODE
create_actwindow;    
shiffu=0.26;
[kx,ky,k,tg1,tg2,tg3,alpha,beta]=freqaxTG(r0,lambda,nx,ny,dx,dy,shiffu);
gcalcf=FW_CH(data,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta);
mapper(x,y,gcalcf,'mGal','Calculated Gravity Model',0,'cntrf')
outmenu_forward(x,y,data,gcalcf)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if modeDT==-1 % INVERSION MODE
[zcalc,gcalc,rmstor,erstor]=maininv_CH_CH(data,nx,ny,dx,dy,r0,lambda,criterio,maxiter,modetermin,wedge);
save('tempinv.mat','zcalc','gcalc','rmstor','erstor','-append')
drawnow;create_actwindow;
mapper(x,y,-zcalc,'Km','Calculated Depth',wedge,'cntrf')
drawnow;set(lstb5,'value',2,'enable','off');
outmenu_inversion(x,y,data,zcalc,gcalc,wedge,rmstor,erstor)
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inversion Procedure
function [zcalc,gcalc,rmstor,erstor]=maininv_CH_CH(go,nx,ny,dx,dy,r0,lambda,criterio,maxiter,modetermin,wedge);
%%%%%%%%%%%%%%%%%%%%%%%%%%wextend anomaly certain grid number wx wy 
%%%%%%%%%%%%%%%%%%%%%%%%%%
shiffu=0.26;
[kx,ky,k,tg1,tg2,tg3,alpha,beta]=freqaxTG(r0,lambda,nx,ny,dx,dy,shiffu);
rmstor=zeros(1,maxiter); %rms vector initial construct
erstor=zeros(1,maxiter); %largest error vector initial construct
const=2*pi*6.673.*r0;
z=-1./lambda.*log(1-(lambda.*go./const));
g=FW_CH(z,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta);% calculated gravity from initial depth
[rms,ers]=rmscalc(go,g,wedge);
rmstor(1)=rms;erstor(1)=ers; %goodness from first approximation
zcalc=z;gcalc=g; % results from first approximation stored (step=1)
step=1;
instaRMSplot(step,rmstor,erstor,modetermin,maxiter,1)
z=(go./gcalc).*z; % iteration starts from step 2
%%%%%%%%%the process will stop according the measure of missfit with the stopping set stopping criteria
%%% or at maxiter 
if modetermin==1;cmpv=rms;critere=criterio;end;
if modetermin==2;cmpv=rms;critere=0;end;
if modetermin==3;cmpv=rms;critere=criterio;end;
if modetermin==4;cmpv=ers;critere=criterio;end;
tic
while cmpv>critere
step=step+1;
g_old=gcalc;
z_old=zcalc;
gcalc=FW_CH(z,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta); %new gcalc
[rms,ers]=rmscalc(go,gcalc,wedge);
% Case stopping mode is RMS divergence
% if true, the results at step-1 are stored and the loop breaks
if modetermin==2;
if rms>rmstor(step-1);step=step-1;gcalc=g_old; zcalc=z_old;break;end;
end
rmstor(step)=rms;
erstor(step)=ers;
instaRMSplot(step,rmstor,erstor,modetermin,maxiter,1)
zcalc=z; %zcalc associated to gcalc
%%%%%%%%%%%%% auto-stopping,checking or resume
lstb5=get(findobj(gcf,'Tag','listb5'),'value');
if lstb5==1;
create_actwindow;
tit=['RMS (step)=' num2str(rms) '   Max.Err (step)=' num2str(ers)];
mapper(1:nx,1:ny,-zcalc,'Km',tit,wedge,'cntrf')
uiwait;
end
lstb5=get(findobj(gcf,'Tag','listb5'),'value');
if lstb5==3;break;end
%%%%%%%%%%%%%
% Case stopping mode is RMS Diff. 
% the difference between successive steps is checked..
% if true the difference is below the threshold it stops...
if modetermin==3;
if abs(rms-rmstor(step-1))<criterio;break;end;
end
if step==maxiter;break;end % maximum number of iteration accomplished
z=(go./gcalc).*zcalc; % modification of z for new iteration if still (cfit>criterio) 
if modetermin==1 || modetermin==2 || modetermin==3; cmpv=rms;end;
if modetermin==4;cmpv=ers;end;
end
toc
rmstor=rmstor(1:step);% the rms at last step
erstor=erstor(1:step);% the largest error at last step
instaRMSplot(step,rmstor,erstor,modetermin,maxiter,0)% view statistics
end

function [kx,ky,k,tg1,tg2,tg3,alpha,beta]=freqaxTG(r0,lambda,nx,ny,dx,dy,shiffu)
shiffv= shiffu;
dkx= 1./((nx).*dx); 
dky= 1./((ny).*dy);
nyqx= (nx/2)+1; % data is even
nyqy= (ny/2)+1; % data is even
kx=[((0:nx/2)+shiffu) (nyqx+1:nx)-(nx+1)+shiffu].*dkx;
ky=[((0:ny/2)+shiffv) (nyqy+1:ny)-(ny+1)+shiffv].*dky;
[ux,uy]=meshgrid(kx,ky);
k=sqrt(ux.^2+uy.^2);
%%%%%%%%%%%%%%%%%%%%%%%%%TG
a=dx/2;b=dy/2;
tg1 = 2.*pi.*6.673.*r0;
tg2 =(4.*a.*b.*sin(pi.*2.*ux.*a)./(pi.*2.*ux.*a)).*(sin(pi.*2.*uy.*b)./(pi.*2.*uy.*b));
tg3 =1./(lambda + 2.*pi.*k);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%alfabeta
alpha = (0:nx-1) * dx;
beta  = (0:ny-1).' * dy;
end

function dg=FW_CH(z,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta)
shiffv= shiffu;
guv= zeros(ny,nx);
for j = 1:ny
for i = 1:nx
tg4 = (1 - exp(-(lambda(j,i) + 2 * pi * k(j,i)) .* z)); 
tg5 = exp(-2 * pi *(1i)*(bsxfun(@plus, kx(i) * alpha, ky(j) .* beta)));
guv(j, i) =tg1(j,i)*tg2(j,i)*tg3(j,i)*(sum(sum(tg4 .* tg5)));
end
end
guv=(ifft2(guv))./(dx*dy);
for j = 1:ny
for i= 1:nx
guv(j,i) =  guv(j,i).*exp( 2.*pi.*(1i).*(((j-1).*shiffv./ny) + ((i-1).*shiffu./nx)));
end
end
dg=real(guv);
end

function [rms,ers]=rmscalc(D1,D2,wedge) %% rms and max.error calculator
 D1=D1(wedge+1:end-wedge,wedge+1:end-wedge);
 D2=D2(wedge+1:end-wedge,wedge+1:end-wedge);
 dg2=(D1-D2).^2;
 rms=sqrt(sum(sum(dg2))./((numel(D1))));
 ers=max(max(abs(D1-D2)));
end
%%%%%%%%%%%%%%%%%%%%%%% GUI related Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function create_MainGui(~,~) %%% MainFigureWindow
fig1 = figure('MenuBar','none','Name','Grav3CH_inv',...
'NumberTitle','off','Resize','off','Tag','mainfig','Color','w',...
'units','normalized','outerposition',[0 0.05 0.35 0.95],'DockControls','off');
ax1=axes('Parent',fig1,'units','normalized','Position',...
[0.1 0.08 0.8 0.35]);axis off
menu1 = uimenu('Parent',fig1,'Label','LOAD DATA');
menu11 = uimenu('Parent',menu1,'Label','INVERSION');
menu12 = uimenu('Parent',menu1,'Label','FORWARD CALC.');
uimenu('Parent',menu11,'Label','New 2-D Gravity Grid (*.grd)','CallBack',{@datainput,-1,ax1});
uimenu('Parent',menu12,'Label','New 2-D Depth Grid (*.grd)','CallBack',{@datainput,1,ax1});
uimenu('Parent',menu1,'Label','OPEN PROJECT (*.mat)','CallBack',{@datainput,0,ax1});
menu2 = uimenu('Parent',fig1,'Label','TOOLS');
uimenu('Parent',menu2,'Label','Exp. Fit Tool','CallBack',@densmodelfit_menu);
uicontrol('Parent',fig1,'Style','listbox','units','normalized','Position',[0,.94,1,.06],...
'String',{'Grid Info:'},'Tag','listb1','BackGroundColor','w')
uicontrol('Parent',fig1,'Style','pushbutton','units','normalized',...
'Position',[0,.9,1,.03],'String','PLOT INPUT GRID','CallBack',{@showmap,0})
%%%% panel settings-1 
pn1=uipanel('Parent',fig1,'units','normalized','Position',[0,.73,1,.15],...
    'title','Settings 1: Density Model','Fontweight','bold','BackgroundColor','w');
uicontrol('Parent',pn1,'Style','text','units','normalized',...
'Position',[0,.69,.29,.2],'String','r0 (density contrast) >>','Fontweight','bold',...
'BackGroundColor','w')
uicontrol('Parent',pn1,'Style','text','units','normalized',...
'Position',[0,.12,.29,.2],'String','lambda (decay rate) >>','Fontweight','bold',...
'BackGroundColor','w')
edr0=uicontrol('Parent',pn1,'Style','edit','units','normalized',...
'Position',[.61,.8,.38,.2],'String','-0.57','Fontweight','bold',...
'BackGroundColor','w','Tag','edr0');
edlambda=uicontrol('Parent',pn1,'Style','edit','units','normalized',...
'Position',[.61,.25,.38,.2],'String','0.25','Fontweight','bold',...
'BackGroundColor','w','Tag','edlambda');
imp11=uicontrol('Parent',pn1,'Style','pushbutton','units','normalized',...
'Position',[.61,.6,.24,.2],'String','Import Grid','CallBack',{@datainput,11,ax1});
shwmp11=uicontrol('Parent',pn1,'Style','pushbutton','units','normalized',...
'Position',[.855,.6,.135,.2],'String','Plot','CallBack',{@showmap,11});
imp22=uicontrol('Parent',pn1,'Style','pushbutton','units','normalized',...
'Position',[.61,.05,.24,.2],'String','Import Grid','CallBack',{@datainput,22,ax1});
shwmp22=uicontrol('Parent',pn1,'Style','pushbutton','units','normalized',...
'Position',[.855,.05,.135,.2],'String','Plot','CallBack',{@showmap,22});
lstb2=uicontrol('Parent',pn1,'Style','listbox','units','normalized','Position',[.3,.62,.3,.37],...
'String',{'Constant';'Variable'},'Tag','listb2','BackGroundColor','w');
lstb3=uicontrol('Parent',pn1,'Style','listbox','units','normalized','Position',[.3,.05,.3,.37],...
'String',{'Constant';'Variable'},'Tag','listb3','BackGroundColor','w');
set(lstb2,'Callback',{@stting,'set1',edr0,imp11,shwmp11})
set(lstb3,'Callback',{@stting,'set1',edlambda,imp22,shwmp22})
set(imp11,'enable','off','Tag','pushimp11');set(shwmp11,'enable','off','Tag','pushshwmp11');
set(imp22,'enable','off','Tag','pushimp22');set(shwmp22,'enable','off','Tag','pushshwmp22');
%%%%%%%%%%%% panel settings-2
pn2=uipanel('Parent',fig1,'units','normalized','Position',[0,.55,1,.13],...
    'title','Settings 2: Iteration Stop','Fontweight','bold','BackgroundColor','w');
uicontrol('Parent',pn2,'Style','text','units','normalized',...
'Position',[0,.7,.29,.25],'String','Select Mode >>','Fontweight','bold',...
'BackGroundColor','w')
lstb4=uicontrol('Parent',pn2,'Style','listbox','units','normalized','Position',[.3,.1,.3,.83],...
'String',{'RMS CONV.';'RMS DIV.';'RMS Diff.';'Max. Err'},'Tag','listb4','BackGroundColor','w');
uicontrol('Parent',pn2,'Style','text','units','normalized',...
'Position',[.02,.33,.25,.25],'String','|| threshold ||','Fontweight','bold',...
'BackGroundColor','w')
edcrit=uicontrol('Parent',pn2,'Style','edit','units','normalized',...
'Position',[.02,.05,.25,.25],'String','0.05','Fontweight','bold',...
'BackGroundColor','w','Tag','edcrit');
set(lstb4,'Callback',{@stting,'set2',edcrit})
uicontrol('Parent',pn2,'Style','text','units','normalized',...
'Position',[.6,.65,.39,.25],'String','maximum number of iterations','Fontweight','bold',...
'BackGroundColor','w')
edmxiter=uicontrol('Parent',pn2,'Style','edit','units','normalized',...
'Position',[.65,.4,.3,.2],'String','30','Fontweight','bold',...
'BackGroundColor','w','Tag','edmxiter');
%%%%%%%%%%%%%%% panel3 ignore data 
pn3=uipanel('Parent',fig1,'units','normalized','Position',[0,.49,.2,.055],...
'title','Ignore Data','Fontweight','bold','BackgroundColor','w',...
'BorderType','none','titleposition','centerbottom');
edign=uicontrol('Parent',pn3,'Style','edit','units','normalized',...
'Position',[0,.2,1,.7],'String','0','Fontweight','bold',...
'BackGroundColor','w','Tag','edign');
lstb5=uicontrol('Parent',fig1,'Style','listbox','units','normalized','Position',[.86,.45,.14,.065],...
'String',{'Pause';'Resume';'Break'},'Tag','listb5','BackGroundColor','w',...
'value',2,'CallBack',@algor_stop);
%%%%%%%%%%%%%%% calculation starter buttons
uicontrol('Parent',fig1,'Style','pushbutton','units','normalized',...
'Position',[.6,.69,.39,.03],'String','Forward Calc.','Tag','frwbut',...
'ForeGroundColor','b','FontWeight','bold','enable','off',...
'CallBack',{@startfunc,1,lstb2,lstb3,lstb4,lstb5,edr0,edlambda,edcrit,edmxiter,edign})
uicontrol('Parent',fig1,'Style','pushbutton','units','normalized',...
'Position',[.2,.515,.8,.03],'String','START INVERSION','Tag','invbut',...
'ForeGroundColor','r','FontWeight','bold','enable','off',...
'CallBack',{@startfunc,-1,lstb2,lstb3,lstb4,lstb5,edr0,edlambda,edcrit,edmxiter,edign})
%%%%%%%%%%%% 
data=0;r0=0;lambda=0;% initialize temporary file content
save('tempinv.mat','data','r0','lambda')
end
function algor_stop(src,~)
val=get(src,'value');
if val==3;set(src,'enable','off');end
if val==2 || val==3;
drawnow;delete(findobj('Type','figure','name','MAP'));
uiresume;
end
end

function figact=create_actwindow %%% create empty window
drawnow;delete(findobj('Type','figure','name','MAP'));
figact=figure('MenuBar','none','Name','MAP','NumberTitle','off','Resize','off',...
'Color','w','units','normalized','outerposition',[.355 .05 .645 .95],...
'DockControls','off');
end

function backmainmenu(~,~)
drawnow;delete(findobj('Type','figure','name','MAP'));
end

function backoutmenu(~,~,x,y,data,zcalc,gcalc,wedge,rmstor,erstor)
drawnow;delete(findobj('Type','figure','name','MAP'));
create_actwindow;
outmenu_inversion(x,y,data,zcalc,gcalc,wedge,rmstor,erstor)
mapper(x,y,-zcalc,'Km','Calculated Depth',wedge,'cntrf')
end

function densmodelfit_menu(~,~)
drawnow;delete(findobj('Type','figure','name','MAP'));
create_actwindow;
uimenu('Parent',gcf,'Label','< Back','ForeGroundColor','r','Callback',@backmainmenu);
m1=uimenu('Parent',gcf,'Label','Import Data','ForeGroundColor','b');
m11=uimenu('Parent',m1,'Label','Load Depth- Density Contrast data (*.txt, *.dat)');
m2=uimenu('Parent',gcf,'Label','Export Image');
uimenu('Parent',m2,'Label','BMP','CallBack',@imageout);
uimenu('Parent',m2,'Label','JPEG','CallBack',@imageout);
uimenu('Parent',m2,'Label','PNG','CallBack',@imageout);
uimenu('Parent',m2,'Label','EMF','CallBack',@imageout);
uimenu('Parent',m2,'Label','TIFF','CallBack',@imageout);
m3=uimenu('Parent',gcf,'Label','','enable','off');
set(m11,'CallBack',{@load_dens_measure,m3})
text(.01,.4,{'Exponential fit tool for calculating r0 and lambda parameters';...
'----------------------------------';'';...
'1-Import two column data from an ascii file [depth vs dens.contrast]';...
'note: depth (km) in negative';...
'      contrast (gr/cc) in negative'});
axis off
end

function outmenu_forward(x,y,data,gcalcf)
m1=uimenu('Parent',gcf,'Label','Export Image');
uimenu('Parent',m1,'Label','BMP','CallBack',@imageout);
uimenu('Parent',m1,'Label','JPEG','CallBack',@imageout);
uimenu('Parent',m1,'Label','PNG','CallBack',@imageout);
uimenu('Parent',m1,'Label','EMF','CallBack',@imageout);
uimenu('Parent',m1,'Label','TIFF','CallBack',@imageout);
m2=uimenu('Parent',gcf,'Label','Switch to Map');
uimenu('Parent',m2,'Label','Calculated Gravity Model',...
'CallBack',{@switch2plot,x,y,gcalcf,'mGal','Calculated Gravity Model',0,'map'});
uimenu('Parent',m2,'Label','Depth Model (Input)',...
    'CallBack',{@switch2plot,x,y,-data,'Km','Depth Model',0,'map'});
m3=uimenu('Parent',gcf,'Label','Export Data');
wedge=0;
datasout(:,:,1)=data;
datasout(:,:,2)=gcalcf;
uimenu('Parent',m3,'Label','Export Input & Output as *.grd',...
'CallBack',{@exportdatas,x,y,datasout,wedge,NaN,NaN,'frw','-grav3CH_Z','-grav3CH_Z2GRV'});
end

function outmenu_inversion(x,y,data,zcalc,gcalc,wedge,rmstor,erstor)
m1=uimenu('Parent',gcf,'Label','Export Image');
uimenu('Parent',m1,'Label','BMP','CallBack',@imageout);
uimenu('Parent',m1,'Label','JPEG','CallBack',@imageout);
uimenu('Parent',m1,'Label','PNG','CallBack',@imageout);
uimenu('Parent',m1,'Label','EMF','CallBack',@imageout);
uimenu('Parent',m1,'Label','TIFF','CallBack',@imageout);
m2=uimenu('Parent',gcf,'Label','Switch to Map');
uimenu('Parent',m2,'Label','Calculated Depth',...
'CallBack',{@switch2plot,x,y,-zcalc,'Km','Calculated Depth',wedge,'map'});
uimenu('Parent',m2,'Label','Inverted Anomaly',...
'CallBack',{@switch2plot,x,y,gcalc,'mGal','Inverted Anomaly',wedge,'map'});
uimenu('Parent',m2,'Label','Anomaly Difference',...
'CallBack',{@switch2plot,x,y,data-gcalc,'mGal','Anomaly Difference',wedge,'map'});
uimenu('Parent',m2,'Label','Observed Anomaly',...
'CallBack',{@switch2plot,x,y,data,'mGal','Observed Anomaly',wedge,'map'});
datall(:,:,1)=data;
datall(:,:,2)=gcalc;
datall(:,:,3)=data-gcalc;
datall(:,:,4)=zcalc;
uimenu('Parent',m2,'Label','Map All',...
'CallBack',{@switch2plot,x,y,datall,NaN,NaN,wedge,'map'});
m3=uimenu('Parent',gcf,'Label','Switch to Graph');
uimenu('Parent',m3,'Label','RMS',...
'CallBack',{@switch2plot,NaN,NaN,rmstor,'RMS',NaN,NaN,'graph'});
uimenu('Parent',m3,'Label','Max.Error',...
'CallBack',{@switch2plot,NaN,NaN,erstor,'Max. Error',NaN,NaN,'graph'});
m4=uimenu('Parent',gcf,'Label','Export Data');
uimenu('Parent',m4,'Label','Export Outputs',...
'CallBack',{@exportdatas,x,y,datall,wedge,rmstor,erstor,'inv','-grav3CH_gobs','-grav3CH_gcalc',...
'-grav3CH_gdiff','-grav3CH_zcalc'});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
uimenu('Parent',m4,'Label','SET and Export Profile Data',...
'CallBack',{@exportdatas,x,y,datall,wedge,rmstor,erstor,'prfd','-grav3CH_gobs','-grav3CH_gcalc',...
'-grav3CH_gdiff','-grav3CH_zcalc'});
uimenu('Parent',gcf,'Label','Colormap','Callback',@clrmpeditor)
uicontrol('Parent',gcf,'style','togglebutton','string','3D/2D','units','normalized',...
'Position',[.92 0.96 0.05 0.04],'value',0,'CallBack',{@toggl_2d3d,x,y,zcalc,wedge})
end

function outmenu_profdata(x,y,data,zcalc,gcalc,wedge,rmstor,erstor)
%%%%%%%%%create menu items for profiles 
uimenu('Parent',gcf,'Label','< Back','ForeGroundColor','r','Callback',...
    {@backoutmenu,x,y,data,zcalc,gcalc,wedge,rmstor,erstor});
ax1=axes('Parent',gcf,'units','normalized','Position',[0.05 0.2 0.35 0.6]);
ax3=axes('Parent',gcf,'units','normalized','Position',[0.5 0.05 0.47 0.4]);
ax2=axes('Parent',gcf,'units','normalized','Position',[0.5 0.55 0.47 0.4]);
set(ax2,'XTickLabel','','YTickLabel','');
set(ax3,'XTickLabel','','YTickLabel','');
uicontrol('Parent',gcf,'style','pushbutton','units','normalized','position',...
[.05 .85 .3 .05],'string','Set Profile ([Start line]: left click, [End line]: right click)',...
'ForeGroundColor','k','FontWeight','bold','Callback',{@getcross,ax1,ax2,ax3,x,y,data,zcalc,gcalc});
m1=uimenu('Parent',gcf,'Label','Export Image');
uimenu('Parent',m1,'Label','BMP','CallBack',@imageout);
uimenu('Parent',m1,'Label','JPEG','CallBack',@imageout);
uimenu('Parent',m1,'Label','PNG','CallBack',@imageout);
uimenu('Parent',m1,'Label','EMF','CallBack',@imageout);
uimenu('Parent',m1,'Label','TIFF','CallBack',@imageout);
drawnow;set(gcf,'CurrentAxes',ax1);
mapper(x,y,-zcalc,'Km','Calculated Depth',wedge,'cntrf');
%%%%%% check for last stored profile data;
if ~isempty(who('-file', 'tempinv.mat', 'Dprof'))
load('tempinv.mat','Dprof');
line(Dprof(:,1),Dprof(:,2),'Color','w','linewidth',3);
%%%%%%%%%%%%%%%%% anomaly fig
profile_plot(ax2,ax3,Dprof(:,3),-Dprof(:,4),Dprof(:,5),Dprof(:,6))
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% plotting functions
function mapper(x,y,matrix,unitt,tit,wedge,mpstyl) %%map viewer
xx=x(wedge+1:end-wedge,wedge+1:end-wedge);
yy=y(wedge+1:end-wedge,wedge+1:end-wedge);
matrixx=matrix(wedge+1:end-wedge,wedge+1:end-wedge);
switch mpstyl
    case 'cntrf'
contourf(xx,yy,matrixx,30);shading flat;rotate3d off;box off;
    case 'pclr'
pcolor(xx,yy,matrixx);rotate3d off;box off;
    case 'd3view'
    surf(xx,yy,matrixx,'FaceColor','interp','EdgeColor','none','FaceLighting','gouraud')
    rotate3d on
    box on        
end
axis equal
axis tight
set(gca,'FontSize',10,'FontWeight','bold','box','on')
h=colorbar('EastOutside');title(h,unitt,'FontWeight','bold');
set(h,'FontSize',10,'FontWeight','bold')
xlabel('X (Km)');ylabel('Y (Km)');title(tit)
xlim([min(min(x)) max(max(x))])
ylim([min(min(y)) max(max(y))])
end

function instaRMSplot(step,rmstor,erstor,modetermin,maxiter,lmt)
drawnow %% instant plot of rms or max. error during on-going iteration
crit=str2double(get(findobj(gcf,'Tag','edcrit'),'string'));
if modetermin==1 
plot(1:step,rmstor(1:step),'-k+');grid on;
hold on;
plot(step,crit,'ro');hold off;
xlim([1 maxiter]);ylim([0 rmstor(1)]);
title('RMS(step) > threshold')
if lmt==0;xlim([1 step]);
str1=['RMS(1)=' num2str(rmstor(1))];
str2=['RMS(end)=' num2str(rmstor(step))];
str3=['RMS(end-1) - RMS(end)=' num2str(rmstor(step-1)-rmstor(step))];
title([{str1} {str2} {str3}],'Color','b')
xlabel(['performed iteration=' num2str(step)])
legend('RMS','Threshold')
end
end
if modetermin==2 
plot(1:step,rmstor(1:step),'-k+');grid on;
xlim([1 maxiter]);ylim([0 rmstor(1)]);
title('RMS(step)< RMS(step-1)')
if lmt==0;xlim([1 step]);
str1=['RMS(1)=' num2str(rmstor(1))];
str2=['RMS(end)=' num2str(rmstor(step))];
str3=['RMS(end-1) - RMS(end)=' num2str(rmstor(step-1)-rmstor(step))];
title([{str1} {str2} {str3}],'Color','b')
xlabel(['performed iteration=' num2str(step)])
end
end
if modetermin==3 
plot(1:step,rmstor(1:step),'-k+');grid on;
xlim([1 maxiter]);ylim([0 rmstor(1)]);
title('RMS(step-1)-RMS(step) > threshold')
if lmt==0;xlim([1 step]);
str1=['RMS(1)=' num2str(rmstor(1))];
str2=['RMS(end)=' num2str(rmstor(step))];
str3=['RMS(end-1) - RMS(end)=' num2str(rmstor(step-1)-rmstor(step))];
title([{str1} {str2} {str3}],'Color','b')
xlabel(['performed iteration=' num2str(step)])
end
end
if modetermin==4;
plot(1:step,erstor(1:step),'-k+');grid on;
hold on;
plot(step,crit,'ro');hold off;
xlim([1 maxiter]);ylim([0 erstor(1)]);
title('Max. ERR(step) > threshold')
if lmt==0;xlim([1 step]);
str1=['ERR(1)=' num2str(erstor(1))];
str2=['ERR(end)=' num2str(erstor(step))];
str3=['ERR(end-1) - ERR(end)=' num2str(erstor(step-1)-erstor(step))];
title([{str1} {str2}],'Color','b')    
xlabel(['performed iteration=' num2str(step)])
legend('RMS','Threshold')
end
end
set(gca,'YTickLabel',[]);
drawnow
end

function graph_plot(errdata,unitstr)
switch unitstr
    case 'RMS'
        clr='r';str1='RMS(1)=';str2='RMS(end)=';
        ylm=[floor(errdata(end)) errdata(1)+0.1];
    case 'Max. Error'
        clr='b';str1='ERR(1)=';str2='ERR(end)=';
        ylm=[floor(errdata(end)) ceil(errdata(1))];
end
plot(errdata,'-ko','MarkerFaceColor',clr,'MarkerSize',5);
set(gca,'FontSize',10,'FontWeight','bold')
str1=[str1 num2str(errdata(1))];
str2=[str2 num2str(errdata(end))];
title ([str1 '    ' str2])
xlabel('Iteration number');ylabel(unitstr);
xlim([1 numel(errdata)]);ylim(ylm);
grid on
set(gca,'FontSize',10,'FontWeight','bold')
end

function mapinfo(T,nx,ny,xmin,xmax,ymin,ymax,dx,dy,fnam) %%% print mesh info of input grid
set(findobj(gcf,'Tag','listb1'),'value',1);
s1=['[Source]:  ' fnam];
s2=['        [Size NX/NY]:  ' num2str(nx) '  /   ' num2str(ny)];
s3=['        [Grid Spacing dx/dy]:  ' num2str(dx) '  /   ' num2str(dy)];
s4=['        [Xmin/Xmax]:  ' num2str(xmin) '  /   ' num2str(xmax)];
s5=['        [Ymin/Ymax]:  ' num2str(ymin) '  /   ' num2str(ymax)];
s6=['        [Zmin/Zmax]:  ' num2str(min(min(T))) '  /  ' num2str(max(max(T)))];
str={[s1 s2 s3 s4 s5 s6]};
set(findobj(gcf,'Tag','listb1'),'string',str);
end

function profile_plot(ax2,ax3,PROFX,PROFZ,PROFG0,PROFGC)
drawnow;set(gcf,'CurrentAxes',ax2);
plot(PROFX,PROFG0,'+r',PROFX,PROFGC,'-k','linewidth',1);grid on;box on;
xlim([min(PROFX) max(PROFX)]);
ylabel('mGal','FontSize',10,'FontWeight','bold');
xlabel('Distance (Km)','FontSize',10,'FontWeight','bold')
legend('Observed','Calculated','Location','SouthEast');
set(gca,'FontSize',10,'FontWeight','bold')
set(ax2,'XTickLabel','')
%%%%%%%%%%%%%%%%% depth model fig
drawnow;set(gcf,'CurrentAxes',ax3);
dfz=max(PROFZ)+(max(PROFZ)/5);
PROFZ(end+1:end+3)=[dfz dfz PROFZ(1)]; 
PROFX(end+1:end+3)=[PROFX(end) PROFX(1) PROFX(1)];
plot(PROFX(1:end-3),-PROFZ(1:end-3),'-k','linewidth',2);
patch(PROFX,-PROFZ,[.1 .1 .1],'FaceAlpha',.8);
ylabel('Depth (Km)','FontSize',10,'FontWeight','bold')
xlim([min(PROFX) max(PROFX)]);
ylim([min(-PROFZ) 0]);
set(ax3,'XAxisLocation','top');
box on
grid on
set(gca,'FontSize',10,'FontWeight','bold')
end

function showmap(~,~,mtyp)
% if mtyp==0 then the mapping is for gravity (-1) or depth (1) input
% if mtyp==11 mapping is r0
% if mtyp==22 mapping is lambda
load('tempinv.mat','data');
if numel(data)>1
switch mtyp
    case 0
        load('tempinv.mat','x','y','modeDT');
        create_actwindow;
        switch modeDT
            case 1
            mapper(x,y,-data,'Km','Depth Model',0,'cntrf')    
            case -1
            mapper(x,y,data,'mGal','Observed Gravity',0,'cntrf')    
        end    
     case 11
        if isempty(who('-file', 'tempinv.mat', 'r02d'));error_message(6);return;end 
        load('tempinv.mat','x','y','r02d');
        if isequal(size(x),size(r02d))==0;error_message(5);return;end
        create_actwindow;
        mapper(x,y,r02d,'g/cc','r0',0,'pclr');
     case 22
        if isempty(who('-file', 'tempinv.mat', 'lambda2d'));error_message(6);return;end 
        load('tempinv.mat','x','y','lambda2d');
        if isequal(size(x),size(lambda2d))==0;error_message(5);return;end
        create_actwindow;
        mapper(x,y,lambda2d,'1/Km','decay rate',0,'pclr');
end
end
end

function switch2plot(src,~,x,y,matrix,unitt,titt,wedge,typ)
switch typ
    case 'map'
        if size(matrix,3)==1;
            ax=findall(gcf,'type','axes');drawnow;delete(ax);
            if strcmp(get(src,'label'),'Calculated Depth');
            drawnow;set(findall(gcf,'style','togglebutton'),'enable','on','value',0);
            else
            drawnow;set(findall(gcf,'style','togglebutton'),'enable','off','value',0);    
            end
            mapper(x,y,matrix,unitt,titt,wedge,'cntrf');
            set(findall(gcf,'type','uimenu','Label','Colormap'),'enable','on');
        end
        if size(matrix,3)==4;
      set(findall(gcf,'type','uimenu','Label','Colormap'),'enable','off');
      drawnow;set(findall(gcf,'style','togglebutton'),'enable','off');
      drawnow;plot(.5,.5,'k+');title('preparing plots...');axis off;drawnow    
      subplot(2,2,1);mapper(x,y,matrix(:,:,1),'mGal','Observed Anomaly',wedge,'cntrf')    
      subplot(2,2,2);mapper(x,y,matrix(:,:,2),'mGal','Inverted Anomaly',wedge,'cntrf')
      subplot(2,2,3);mapper(x,y,matrix(:,:,3),'mGal','Anomaly Difference',wedge,'cntrf')
      subplot(2,2,4);mapper(x,y,-matrix(:,:,4),'Km','Calculated Depth',wedge,'cntrf')    
        end    
    case 'graph'
        drawnow;set(findall(gcf,'type','uimenu','Label','Colormap'),'enable','off');
        drawnow;set(findall(gcf,'style','togglebutton'),'enable','off');
        ax=findall(gcf,'type','axes');drawnow;delete(ax);
        graph_plot(matrix,unitt)
end
end

function error_message(e) %%% create error message window
drawnow;delete(findobj('Type','figure','name','MAP'));
figure('MenuBar','none','Name','MAP',...
'NumberTitle','off','Resize','off',...
'Color','k',...
'units','normalized','outerposition',[0.355 .7 .645 .1],...
'DockControls','off');
switch e
    case 1
        text(0,.5,'density contrast can not be zero !...please set','Color','w')
    case 2
        text(0,.5,'equal spaced grid dx=dy is required !','Color','w')
    case 3
 text(0,.5,'File format not supported..Load Surfer 6 text grid / Surfer 7 Binary Grid','Color','w')
    case 4
 text(0,.5,'Blanked grid not supported !','Color','w')
    case 5
 text(0,.5,'Size missmatch between DATA grid and 2-D density model grid !','Color','w');
    case 6
 text(0,.5,'2-D Density Model not available !...please SET','Color','w');
    case 7
 text(0,.5,'Please set the iteration stopping criteria ...','Color','w');
    case 101
 text(0,.5,'Input Data not available !...please Load...','Color','w')       
end
axis off
end

function clrmpeditor(~,~)
colormapeditor
end

function toggl_2d3d(src,~,x,y,zcalc,wedge)
val=get(src,'value');
if val==1;mapper(x,y,-zcalc,'Km','Calculated Depth',wedge,'d3view');daspect([1 1 .25]);end
if val==0;mapper(x,y,-zcalc,'Km','Calculated Depth',wedge,'cntrf');end
end
%%%%%%%%%%%%%%%%%%%%%% Data Exporting functions...........
function exportdatas(~,~,x,y,datasout,wedge,rmstor,erstor,typ,ext1,ext2,ext3,ext4)
%%%% export maps as *.grd and graphics as *.dat
if strcmp(typ,'inv') || strcmp(typ,'frw')
[filename, pathname] = uiputfile('*.grav3CH','Set Fore-name of files');
if ischar([pathname filename])
mkdir(pathname,[filename(1:end-8) '-grav3CH_project']);    
kk=[pathname filename(1:end-8) '-grav3CH_project\' filename(1:end-8)];
xmin=min(min(x(wedge+1:end-wedge,wedge+1:end-wedge)));
xmax=max(max(x(wedge+1:end-wedge,wedge+1:end-wedge)));
ymin=min(min(y(wedge+1:end-wedge,wedge+1:end-wedge)));
ymax=max(max(y(wedge+1:end-wedge,wedge+1:end-wedge)));
else
    return
end
end

switch typ
    case 'frw'
    MZ=datasout(:,:,1); %% depth model
    MZ=MZ(wedge+1:end-wedge,wedge+1:end-wedge);
    MGRV=datasout(:,:,2); %% gravity model
    MGRV=MGRV(wedge+1:end-wedge,wedge+1:end-wedge);
    grdout(-MZ,xmin,xmax,ymin,ymax,[kk ext1 '.grd']);
    grdout(MGRV,xmin,xmax,ymin,ymax,[kk ext2 '.grd']);
    case 'inv'
    gobs=datasout(:,:,1);gobs=gobs(wedge+1:end-wedge,wedge+1:end-wedge);
    gcalc=datasout(:,:,2);gcalc=gcalc(wedge+1:end-wedge,wedge+1:end-wedge);
    gdiff=datasout(:,:,3);gdiff=gdiff(wedge+1:end-wedge,wedge+1:end-wedge);
    zcalc=datasout(:,:,4);zcalc=zcalc(wedge+1:end-wedge,wedge+1:end-wedge);
    grdout(gobs,xmin,xmax,ymin,ymax,[kk ext1 '.grd']);
    grdout(gcalc,xmin,xmax,ymin,ymax,[kk ext2 '.grd']);
    grdout(gdiff,xmin,xmax,ymin,ymax,[kk ext3 '.grd']);
    grdout(-zcalc,xmin,xmax,ymin,ymax,[kk ext4 '.grd']);
    matrix=[(1:numel(rmstor))' rmstor' erstor'];
    save([kk '-grav3CH_rmserr.dat'],'matrix','-ascii');
    copyfile('tempinv.mat', [kk '-grav3CH_project.mat']);
    case 'prfd'
    drawnow;delete(findobj('Type','figure','name','MAP'));
    fg=create_actwindow;
    set(fg,'outerposition',[0.1 0.1 .9 .9]);
    outmenu_profdata(x,y,datasout(:,:,1),datasout(:,:,4),datasout(:,:,2),wedge,rmstor,erstor)
end
end

function getcross(~,~,ax1,ax2,ax3,x,y,data,zcalc,gcalc)
drawnow;set(gcf,'CurrentAxes',ax1);
cla(ax2,'reset');
cla(ax3,'reset');
delete(findall(gcf,'type','line'));
delete(findobj(gcf,'Tag','savprofbut'));
[xL,yL]=getline(ax1,2);
if length(xL)>1
xL=[xL(1) xL(end)];yL=[yL(1) yL(end)];
xc=linspace(xL(1),xL(2),100);yc=linspace(yL(1),yL(2),100);
disx=abs(xL(1)-xL(2));disy=abs(yL(1)-yL(2));PL=hypot(disx,disy);
PROFX=linspace(0,PL,100);% x coordinates of profile 
PROFZ=interp2(x,y,zcalc,xc,yc); % profile of depth
PROFG0=interp2(x,y,data,xc,yc); % profile of observed
PROFGC=interp2(x,y,gcalc,xc,yc); % profile of observed
Dprof=[xc' yc' PROFX' -PROFZ' PROFG0' PROFGC'];
line(xc,yc,'Color','w','linewidth',3);
%%%%%%%%%%%%%%%%% anomaly fig
profile_plot(ax2,ax3,PROFX,PROFZ,PROFG0,PROFGC)
uicontrol('Parent',gcf,'style','pushbutton','units','normalized','position',...
    [0.05 0.1 0.3 0.04],'string','Export Profile Data (xmap,ymap,xprof,z,gobs,gcalc)',...
    'Tag','savprofbut','ForegroundColor','r','CallBack',{@savDprof,Dprof})
end
end

function savDprof(src,~,Dprof);
[filename, pathname] = uiputfile('*.dat','Set Output File Name');
if ischar([pathname filename])
save([pathname filename],'Dprof','-ascii');
save('tempinv.mat','Dprof','-append');
set(src,'string','Profile Data STORED','ForegroundColor','b')
end
end
function imageout(src,~) %capture screen as image 
set(gcf, 'PaperPositionMode','auto')
srcLb=get(src,'Label');
switch srcLb
    case 'BMP' 
    ext='-dbmp';flt='*.bmp';
    case 'JPEG'
    ext='-djpeg';flt='*.jpg';
    case 'PNG'
    ext='-dpng';flt='*.png';
    case 'EMF'
    ext='-dmeta';flt='*.emf';  
    case 'TIFF'
    ext='-dtiff';flt='*.tiff';
end
[filename, pathname] = uiputfile(flt,'Set a filename');
trgtf=[pathname filename];
if ischar(trgtf)
drawnow
print(trgtf,ext,'-noui','-r300');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%% GUI settings
function stting (src,~,setmod,edx,impx,shwmpx)
switch setmod
    case 'set1'
    drawnow;delete(findobj('Type','figure','name','MAP'));
    val=get(src,'value');
    if val==1; 
    set(edx,'enable','on');set(impx,'enable','off');set(shwmpx,'enable','off');
    else
    set(edx,'enable','off');set(impx,'enable','on');set(shwmpx,'enable','on'); 
    end
    case 'set2'
    val=get(src,'value');   
    if val==1 || val==3 || val==4;set(edx,'enable','on');else;set(edx,'enable','off');end     
end
end

function resetmainwindow
set(findobj(gcf,'Tag','listb2'),'enable','on')
set(findobj(gcf,'Tag','listb3'),'enable','on')
set(findobj(gcf,'Tag','listb4'),'enable','on')
val2=get(findobj(gcf,'Tag','listb2'),'value');
val3=get(findobj(gcf,'Tag','listb3'),'value');
if val2==1
set(findobj(gcf,'Tag','pushimp11'),'enable','off')
set(findobj(gcf,'Tag','pushshwmp11'),'enable','off')
else
set(findobj(gcf,'Tag','pushimp11'),'enable','on')
set(findobj(gcf,'Tag','pushshwmp11'),'enable','on')    
end
if val3==1
set(findobj(gcf,'Tag','pushimp22'),'enable','off')
set(findobj(gcf,'Tag','pushshwmp22'),'enable','off')
else
set(findobj(gcf,'Tag','pushimp22'),'enable','on')
set(findobj(gcf,'Tag','pushshwmp22'),'enable','on')    
end
end