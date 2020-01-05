function Grav3CH_inv
clc;clear all;clear global;
%Code by 
%Oksum. E. : eroksum@gmail.com / erdincoksum@sdu.edu.tr 
% 2019
%%%%%%%%%%%%%%%%%%%
create_MainGui
%%%%%%%%%%%%%%%%%%%% Brief Descriptions %%%%%%%%%
%%%%%%%%% INVERSION 
     % 1- Import gridded gravity data [2-D grid (*.grd]
     %    by Load Data menu at main gui
     % 2- Set Density contrast mode Vertical (default) or to Vertical and Horizontal
     %    by radio button selection. 
          % Case Vertical: Decay constant parameter is single value 
          %                (equal for all prisms)
          % Case Vertical and Horizontal: Decay constant parameter is 2-D
          % grid required to be in same size of input gravity grid...
          % A pushbutton "Set" appears in the table content allowing input of this 
          % required 2-D data by a file read..
     % 3- Set r0 (surface density contrast) and decay constant 
          %(single value or 2-D map) to the related fields of table input.
     % 4- Set termination criterion and maximum number of iteration to 
          % the related cells of "termin" table 
     % 3- Start Iteration..
     
     % View Outputs : Basement depth, Inverted anomaly, Anomaly Difference, 
     %                RMS plot, Max.error plot. 
     %                (interactively selectable by a listbox activated after 
     %                mouse click on outputs GUI)
     % View Options : 2D, 3D (only for depth model) 
     %                Cross-Section (from depth,observed and calculated anomalies)
     %                [Cross-section can be performed only in 2-D view of maps]  
     % Export Outputs: Image file (*.png, 300 dpi resolution),
     %                 Data file (maps-> *.grd, graphics-> *.dat)
     %                 Export as project file (*CHproj.mat). Project files
     %                 can be loaded any time for re-view of interpretation
     %
%%%%%%%%% FORWARD CALCULATION
     %    1- Import gridded depth data [2-D grid (*.grd)] by Tools/Forward menu 
     %    2- Set density contrast and decay constant model (single or 2-D)
     %    3- Start Forward   
     %    4- Export (image file/data file)
     
%%%%%%%%%%%%%%% Tasks of some main Functions
%%% impgrd > NEW 2-D GRID and initialize/memorize parameters
             % uses grd2loader,lodgrd6txt,lodgrd7bin
%%% startiter > retrievs inputs, performs inversion/forward
             % memorize outputs
%%% maininv_0CHCH > performs inversion sheme
%%% FW_CH    > performs forward calculation
%%% freqaxTG > calculates wavenumbers k, kx,ky and variables standing outside 
             % the loop of forward  

%%% please see Manual pdf file for the tasks of all functions used
%%% in code
end
function create_MainGui(~,~) %%% MainFigureWindow
fig1 = figure('MenuBar','none','Name','Grav3CH_inv',...
'NumberTitle','off','Resize','off',...
'Tag','mainfig','Color','w',...
'units','normalized','outerposition',[0 .1 .25 .9],...
'DockControls','off','WindowButtonDownFcn',@posfig1);
menu1 = uimenu('Parent',fig1,'Label','LOAD DATA');
uimenu('Parent',menu1,'Label','New 2-D Gravity Grid (*.grd)','CallBack',@algor_inv_forw);
uimenu('Parent',menu1,'Label','OPEN PROJECT (*.mat)','CallBack',@algor_inv_forw);
menu2=uimenu('Parent',fig1,'Label','Tools');
menu21=uimenu('Parent',menu2,'Label','Forward Model');
uimenu('Parent',menu21,'Label','New 2-D Depth Grid (*.grd)','CallBack',@algor_inv_forw);
uimenu('Parent',menu2,'Label','Exp.fit Density model','CallBack',@densmodelfit);
%%%%%%%%%%%% Grid info listbox
uicontrol('Parent',fig1,'Style','listbox','units','normalized','Position',[.01,.83,.98,.16],...
'String',{'Grid Info:'},'Tag','listb1','BackGroundColor','w')
%%%%%%%%%%%% MODE selection 
bg=uibuttongroup(fig1,'units','normalized','Position',[0.01 0.77 0.979 0.06],...
'Title','Density Variation Mode','TitlePosition','centertop',...
'FontWeight','bold','SelectionChangeFcn',@radbehav);
%%%radiobuttons in button group
uicontrol(bg,'Style','radiobutton','String','Vertical','FontWeight','bold',...
'units','normalized','Position',[.04 .1 .4 .8],'Tag','rbut1');
uicontrol(bg,'Style','radiobutton','String','Vertical & Horizontal',...
'FontWeight','bold','units','normalized', 'Position',[.45 .1 .45 .8],'Tag','rbut2');
%%%%%%%%%%%% density settings menu
uitable('Parent',fig1,'units','normalized','Position',[.01,.695,.98,.07],...
'ColumnName',{'r0','Decay const.','Ignore Data'},...
'Tag','tabl1','Rowname',{'INV-M'},'Data',[-0.0 0.0 0],'ColumnEditable',true,...
'CellEditCallback',@tabl1_edit);
%%%%%%%%%%%%% Iteration stop criteria menu
uitable('Parent',fig1,'units','normalized','Position',[.01,.61,.98,.08],...
'ColumnName',{'Max-error','Rms-Conv','Maxiter'},...
'Tag','tabl2','Rowname',{'Termin.'},...
'Data',{false,false,30},'ColumnEditable',[false false true],...
'CellSelectionCallback',@termino);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%startiteration
uicontrol('Parent',fig1,'Style','pushbutton','units','normalized',...
'Position',[.01,.56,.98,.04],...
'String','Gravity >> START ITERATION >> Basement','Tag','startbut','ForeGroundColor','b',...
'FontWeight','bold','CallBack',@startiter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%set new temp.file
newd=0;
save ('tempinv.mat','newd')
%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function algor_inv_forw(src,~) %%% data call
srclb=get(src,'Label');
switch srclb
    case 'OPEN PROJECT (*.mat)'
    openproj    
    case 'New 2-D Gravity Grid (*.grd)'
    impgrd(-1)
    case 'New 2-D Depth Grid (*.grd)'
    impgrd(1)    
end
end

function impgrd(modeDT) %% import new grid (gravity or depth)
drawnow
[filename, pathname] = uigetfile('*.grd', 'Import Golden Software Binary/Text grid (*.grd)');
sourcfil=[pathname filename];
if ischar(sourcfil)
[data,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy,err]=grid2checker(sourcfil);
if err>0;error_message(err);return;end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% setting to default
closwindows(12);delaxes;set(findobj(gcf,'Tag','rbut1'),'value',1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% new axes for display input grid
axes('Parent',gcf,'units','normalized','Position',...
[0.02 0.08 0.96 0.35],'Tag','axsig');axis off
%%%%%%
if modeDT==-1
unitstr='mGal';tit='Observed Anomaly';mapsty=0;
mpp=data;
set(findobj(gcf,'Tag','tabl1'),'Rowname',{'INV-M'},'ColumnEditable',true,...
    'Data',[0 0 0]);
set(findobj(gcf,'Tag','tabl2'),'enable','on');
set(findobj(gcf,'Tag','tabl3'),'enable','on');
set(findobj(gcf,'Tag','startbut'),'String','Gravity >> START ITERATION >> Basement')
else
unitstr='km';tit='Depth Model';mapsty=11;
data=(abs(data));% depth is always positive  
mpp=-data; % depth is negative only for illustration
set(findobj(gcf,'Tag','tabl1'),'Rowname',{'FWR-M'},'ColumnEditable',[true true false],...
    'Data',[0 0 0])
set(findobj(gcf,'Tag','tabl2'),'enable','off');
set(findobj(gcf,'Tag','tabl3'),'enable','off');
set(findobj(gcf,'Tag','startbut'),'String','Depth Model >> START Forward >> Gravity')
end
%%%%%%
[~,filnam] = fileparts(sourcfil);
list1w(data,nx,ny,xmin,xmax,ymin,ymax,dx,dy,sourcfil);
save('tempinv.mat','data','x','y','nx','ny','xmin','xmax','ymin','ymax','dx','dy',...
'filnam','modeDT');
drawnow
mapper(x,y,mpp,unitstr,tit,mapsty,2,0)
axis off
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function startiter(~,~)  %%% retrievs inputs, settings, calls inversion
                         %%% or forward, memorize outputs
closwindows(102); 
%%%% CHECK DATA IS AVAILABLE, CHECK DENSITY CONTRAST MODE
nlist=size(get(findobj(gcf,'Tag','listb1'),'string'));
if nlist(1)==1;return;end
modeDC=get(findobj(gcf,'Tag','rbut1'),'value');
if modeDC==0
if isempty(who('-file', 'tempinv.mat', 'lambda2d'));
    error_message(6)
    return;
end
end
%%% Get grid data, mesh info and algorithm type (forward,inversion) 
load('tempinv.mat','x','y','data','nx','ny','dx','dy','modeDT')
%%%%%%%%%%%%%%%%%%%%%%% GET INPUTS %%%%%%%%%%%%%%%%
t1=get(findobj(gcf,'Tag','tabl1'),'Data');
t2=get(findobj(gcf,'Tag','tabl2'),'Data');
if sum([cell2mat(t2(1)) cell2mat(t2(2)) modeDT])<0;error_message(7);return;end
t3=get(findobj(gcf,'Tag','tabl3'),'Data');
r0=t1(1);lambda=t1(2);wedge=t1(3); if lambda==0;lambda=0.001;end
termin=cell2mat(t2(1));
maxiter=cell2mat(t2(3));
criterio=t3;
if r0==0;error_message(1);return;end % stop if density contrast is zero
%%%%%%%%%%%%%%%%%%%%%%%%% store current input settings
save('tempinv.mat','r0','lambda','modeDC','termin','maxiter','criterio','wedge','-append')
%%%%%%%%set lambda to 2D matrix
if modeDC==1;
    lambda=ones(ny,nx)*lambda;
else
    load('tempinv.mat','lambda2d')
    lambda=lambda2d;
end
drawnow
set(findobj(gcf,'Tag','startbut'),'ForeGroundColor','r');
drawnow
%%%%%%%%%%%% PERFORM INVERSION AND/OR FORWARD ALGORITHM %%%%%%%%%%%%%%%%
if modeDT==-1 % inversion mode
delaxes;% clean axes    
[zcalc,gcalc,rmstor,erstor]=maininv_0CHCH(data,nx,ny,dx,dy,r0,lambda,criterio,maxiter,termin,wedge);
save('tempinv.mat','zcalc','gcalc','rmstor','erstor','-append')
mpp=-zcalc;unitstr='Km';tit='Calculated Depth';mapsty=1;
messgstr='Click on screen to switch between Output Plots';
else % forward mode
shiffu=0.26;
[kx,ky,k,tg1,tg2,tg3,alpha,beta]=freqaxTG(r0,lambda,nx,ny,dx,dy,shiffu);
gcalcf=FW_CH(data,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta);
save('tempinv.mat','gcalcf','-append')
mpp=gcalcf;unitstr='mGal';tit='Calculated Gravity Model';mapsty=22;wedge=0;
messgstr='Click on screen to switch between Input/Output';
end
drawnow;set(findobj(gcf,'Tag','startbut'),'ForeGroundColor','b');
figact1=create_actwindow; % empty window
create_outputmenu(figact1,modeDT)
mapper(x,y,mpp,unitstr,tit,mapsty,2,wedge)
xlabel(messgstr,'Color','y','BackGroundColor','k')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inversion Procedure
function [zcalc,gcalc,rmstor,erstor]=maininv_0CHCH(go,nx,ny,dx,dy,r0,lambda,criterio,maxiter,termin,wedge);
%%%%%%%%%%%%%%%%%%%%%%%%%%wextend anomaly certain grid number wx wy 
%%%%%%%%%%%%%%%%%%%%%%%%%%
shiffu=0.26;
[kx,ky,k,tg1,tg2,tg3,alpha,beta]=freqaxTG(r0,lambda,nx,ny,dx,dy,shiffu);
rmstor=zeros(1,maxiter); %rms vector initial construct
erstor=zeros(1,maxiter); %largest error vector initial construct
const=2*pi*6.673*r0;
z=-1./lambda.*log(1-(lambda.*go./const));
g=FW_CH(z,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta);% calculated gravity from initial depth
[rms,ers]=rmscalc(go,g,wedge);
rmstor(1)=rms;erstor(1)=ers; %goodness from first approximation
zcalc=z;gcalc=g; % results from first approximation stored (step=1)
step=1;
axes('Parent',gcf,'units','normalized','Position',...
[0.02 0.08 0.96 0.35],'Tag','axsig');axis off
instaRMSplot(step,rmstor,erstor,termin,maxiter,1)
z=(go./gcalc).*z; % iteration starts from step 2
%%%%%%%%%the process will stop according to goodness of choise or at
%%%%%%%%%maxiter
if termin==0;cfit=rms;end;
if termin==1;cfit=ers;end;
tic
while cfit>criterio
step=step+1;
gcalc=FW_CH(z,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta); %new gcalc
[rms,ers]=rmscalc(go,gcalc,wedge);
rmstor(step)=rms;
erstor(step)=ers;
instaRMSplot(step,rmstor,erstor,termin,maxiter,1)
zcalc=z; %zcalc associated to gcalc
if step==maxiter;break;end % maximum number of iteration accomplished
z=(go./gcalc).*zcalc; % modification of z for new iteration if still (cfit>criterio) 
if termin==0;cfit=rms;end;
if termin==1;cfit=ers;end;
end
toc
rmstor=rmstor(1:step);% the rms at last step
erstor=erstor(1:step);% the largest error at last step
instaRMSplot(step,rmstor,erstor,termin,maxiter,0)% view statistics
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% forward calculation of grav anomalies
function dg=FW_CH(z,lambda,nx,ny,dx,dy,shiffu,kx,ky,k,tg1,tg2,tg3,alpha,beta)
shiffv= shiffu;
guv= zeros(ny,nx);
for j = 1:ny
for i = 1:nx
tg4 = (1 - exp(-(lambda(j,i) + 2 * pi * k(j,i)) .* z)); 
tg5 = exp(-2 * pi *(1i)*(bsxfun(@plus, kx(i) * alpha, ky(j) .* beta)));
guv(j, i) =tg1*tg2(j,i)*tg3(j,i)*(sum(sum(tg4 .* tg5)));
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

function radbehav(~,~) %%% setting density contrast mode and related menu items 
%%%check gravity data is active, otherwise decline action
nlist=size(get(findobj(gcf,'Tag','listb1'),'string'));
if nlist(1)==1;set(findobj(gcf,'Tag','rbut1'),'value',1);return;end
mode=get(findobj(gcf,'Tag','rbut1'),'value');
switch mode
    case 1
        closwindows(1)
    case 0
    if ~isempty(who('-file', 'tempinv.mat', 'filnam_lambda2d'))
    clr='g';strT='View Model';else;clr='y';strT='SET Model';
    end
        uicontrol('Parent',gcf,'Style','pushbutton','units','normalized',...
       'Position',[.45,.706,.225,.03],'String',strT,'FontWeight','bold',...
       'BackGroundColor',[.6 .6 .6],'ForeGroundColor',clr,'Tag','nol',...
       'Callback',@decaymodel_wind)
end
end

function decaymodel_wind(~,~) % temporary window for 2-D decay data 
closwindows(10);
[fig2]=create_actwindow;
set(fig2,'outerposition',[0.255 .55 .35 .45],'Name','Decay-Model')
stringT='Import Decay Constant Grid (*.grd)';
uimenu('Parent',fig2,'Label',stringT,'CallBack',@imporlambda);
checklambda(fig2)
end

function checklambda(fig2) %%% check 2-D decay constant data and display
if ~isempty(who('-file', 'tempinv.mat', 'filnam_lambda2d'))
load('tempinv.mat','x','y','lambda2d','filnam_lambda2d')
mapper(x,y,lambda2d,'km-1',filnam_lambda2d,33,4,0);axis off
else
text(0,.8,'Import a grid file for the horizontal model of decay constant.')
axis off
end
end

function imporlambda (~,~) %%% import 2-D decay constant data, memorize
clc;clear all;clear global;
closwindows(0)
[filename, pathname] = uigetfile('*.grd', 'Import Golden Software Binary/Text grid (*.grd)');
sourcfil=[pathname filename];
if ischar(sourcfil)
[lambda2d,x,y,nx,ny,~,~,~,~,~,~,err]=grid2checker(sourcfil);    
load('tempinv.mat','data');[nydata,nxdata]=size(data);
if nx~=nxdata && ny~=nydata;err=5;end
if err>0;error_message(err);return;end
if ~isempty(who('-file', 'tempinv.mat', 'zcalc'));delaxes;end
if ~isempty(who('-file', 'tempinv.mat', 'gcalc'));delaxes;end
closwindows(2)
%%%%%%%%%%%%%
lambda2d=abs(lambda2d);%positive
lambda2d(find(lambda2d==0))=0.001;
[~,filnam_lambda2d] = fileparts(sourcfil);
save('tempinv.mat','lambda2d','filnam_lambda2d','-append');
drawnow
mapper(x,y,lambda2d,'km-1',filnam_lambda2d,33,4,0);axis off
set(findobj(findobj('Type','figure','name','Grav3CH_inv'),'Tag','nol'),...
'ForeGroundColor','g','String','View Model');
end
end

function closwindows(wnd) % closes windows opened (except the main GUI)
delete(findobj('Type','figure','name','err')); 
delete(findobj('Type','figure','name','Model')); 
switch wnd
    case 1
    delete(findobj('Type','figure','name','Decay-Model'));
    delete(findobj(gcf,'Tag','nol'));
    case 10
    delete(findobj('Type','figure','name','Decay-Model'));
    case 2
    delete(findobj('Type','figure','name','Outputs'));
    case 102
    delete(findobj('Type','figure','name','Decay-Model'));
    delete(findobj('Type','figure','name','Outputs'));
    case 12
    delete(findobj('Type','figure','name','Decay-Model'));
    delete(findobj('Type','figure','name','Outputs'));
    delete(findobj(gcf,'Tag','nol'));
    drawnow
    posfig1
end
end

function delaxes % delete axes for next plot
drawnow
delete(findall(findobj('Type','figure','name','Grav3CH_inv'),'type','axes'));
delete(findall(findobj('Type','figure','name','Decay-Model'),'type','axes'));
delete(findall(findobj('Type','figure','name','Outputs'),'type','axes'));
end

function tabl1_edit(~,~) % setting table content ro,lambda,cutt-off
setto=get(findobj(gcf,'Tag','tabl1'),'Data');
setto=abs(setto);setto(1)=-setto(1);setto(3)=round(setto(3));
set(findobj(gcf,'Tag','tabl1'),'Data',setto);
end


function figact1=create_actwindow %%% create empty window
figact1=figure('MenuBar','none',...
'NumberTitle','off','Resize','off',...
'Color','w',...
'units','normalized','outerposition',[0.255 .1 .745 .9],...
'DockControls','off');
end

function create_outputmenu(figact1,modeDT) %%% create menu item for output window
m1=uimenu('Parent',figact1,'Label','Export Image');
uimenu('Parent',m1,'Label','BMP','CallBack',@imageout);
uimenu('Parent',m1,'Label','JPEG','CallBack',@imageout);
uimenu('Parent',m1,'Label','PNG','CallBack',@imageout);
uimenu('Parent',m1,'Label','EMF','CallBack',@imageout);
uimenu('Parent',m1,'Label','TIFF','CallBack',@imageout);
m2=uimenu('Parent',figact1,'Label','Export Data');
uimenu('Parent',m2,'Label','Export Screen-Data','CallBack',@datasout);
switch modeDT
    case -1
     uimenu('Parent',m2,'Label','Export All-Data','CallBack',@datasout);
     uimenu('Parent',m2,'Label','Save-Project','CallBack',@datasout);
     m3=uimenu('Parent',figact1,'Label','View');
     uimenu('Parent',m3,'Label','View-3D','CallBack',@plotcont);
     uimenu('Parent',m3,'Label','View-2D','CallBack',@plotcont);
     uimenu('Parent',m3,'Label','Set Colormap','CallBack',@plotcont);
     uimenu('Parent',m3,'Label','Cross-Section','Tag','crossb','CallBack',@plotcont);
     set(figact1,'WindowButtonDownFcn',@selectionout_inv,'Name','Outputs')
     case 1
     set(figact1,'WindowButtonDownFcn',@selectionout_forw,'Name','Outputs')
end
end

function error_message(e) %%% create error message window
figure('MenuBar','none','Name','err',...
'NumberTitle','off','Resize','off',...
'Color','k',...
'units','normalized','outerposition',[0.255 .7 .40 .1],...
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
 text(0,.5,'Size of input is not equal to size of observed/model data!','Color','w');
    case 6
 text(0,.5,'2-D Decay Constant Model not available !...please SET','Color','w');
    case 7
 text(0,.5,'Please set the iteration stopping criteria ...','Color','w');
    case 101
 text(0,.5,'Input Data not available !...please Load...','Color','w')       
end
axis off
end

function openproj(~,~) %%% load a project file, created by grav3CH_inv
clc;clear all;clear global;
[filename, pathname] = uigetfile('*-CHproj.mat', 'Open Poject file');
kk=[pathname filename];
if ischar(kk)
%%%%close figures and delete axes, create axes for observed gravity
closwindows(12);delaxes;
axes('Parent',gcf,'units','normalized','Position',...
[0.02 0.08 0.96 0.35],'Tag','axsig');axis off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
copyfile(kk,'tempinv.mat');
load('tempinv.mat');
list1w(data,nx,ny,xmin,xmax,ymin,ymax,dx,dy,kk); %fill info listbox
if modeDC==0;
    set(findobj(gcf,'Tag','rbut2'),'value',1)
else
    set(findobj(gcf,'Tag','rbut1'),'value',1)
end
radbehav
set(findobj(gcf,'Tag','tabl1'),'Rowname',{'INV-M'},'ColumnEditable',true,...
'Data',[r0 lambda wedge]);
tab3=findobj(gcf,'Tag','tabl3');delete(tab3);
set(findobj(gcf,'Tag','startbut'),'Position',[.01,.56,.98,.04])
tbl=uitable(gcf,'units','normalized','Position',[.01,.53,.98,.065],...
'ColumnName',{'current value              '},'Tag','tabl3',...
'Data',criterio,'ColumnEditable',[true],'CellEditCallback',@setcriterio);
 if termin==0
 set(findobj(gcf,'Tag','tabl2'),'Data',{false,true, (maxiter)})
 set(tbl,'Rowname',{'set min.Rms'})
 else
 set(findobj(gcf,'Tag','tabl2'),'Data',{true,false, (maxiter)})
 set(tbl,'Rowname',{'set Max.err'})
 end
 set(findobj(gcf,'Tag','startbut'),'Position',[.01,.48,.98,.04],...
     'String','Gravity >> START ITERATION >> Basement')
 set(findobj(gcf,'Tag','tabl2'),'enable','on');
 set(findobj(gcf,'Tag','tabl3'),'enable','on');
 mapper(x,y,data,'mGal',filnam,0,2,wedge)
 %%open gui for display outputs
 figact1=create_actwindow;
 create_outputmenu(figact1,-1)
 drawnow;plot(.5,.5,'k+');title('preparing plots...');axis off;drawnow    
 subplot(2,2,1);mapper(x,y,data,'mGal','Observed Anomaly',0,2,wedge)    
 subplot(2,2,2);mapper(x,y,gcalc,'mGal','Inverted Anomaly',2,2,wedge)
 subplot(2,2,3);mapper(x,y,data-gcalc,'mGal','Anomaly Difference',3,2,wedge)
 subplot(2,2,4);mapper(x,y,-zcalc,'Km','Calculated Depth',1,2,wedge)
 set(findobj(gcf,'Tag','crossb'),'enable','off')
 xlabel('click on screen to switch between output plots','color','r')
end
end

function plotcont(src,~) %%% view style menu actions
srcLb=get(src,'Label');
load('tempinv.mat','mapsty','x','y','zcalc','gcalc','data','wedge');
if mapsty<4
switch srcLb
    case 'Set Colormap'
        colormapeditor
    case 'View-3D'
      if mapsty==1;mapper(x,y,-zcalc,'Km','Calculated Depth',1,3,wedge);end
      if mapsty==2;mapper(x,y,gcalc,'mGal','Inverted Anomaly',2,3,wedge);end
      if mapsty==0;mapper(x,y,data,'mGal','Observed Anomaly',0,3,wedge);end
      set(findobj(gcf,'Tag','crossb'),'enable','off')
    case 'View-2D'
      if mapsty==1;mapper(x,y,-zcalc,'Km','Calculated Depth',1,2,wedge);end
      if mapsty==2;mapper(x,y,gcalc,'mGal','Inverted Anomaly',2,2,wedge);end
      if mapsty==0;mapper(x,y,data,'mGal','Observed Anomaly',0,2,wedge);end
      set(findobj(gcf,'Tag','crossb'),'enable','on')
    case 'Cross-Section'
        getcross
end
end
end

function getcross % extract profile data
xL=xlabel('Start Line: left mouse click    End Line: right mouse click');
set(xL,'Color','w','FontWeight','Bold','BackGroundColor','k')
load('tempinv.mat','x','y','data','gcalc','zcalc','wedge');
[xxx,yyy]=getline;
if length(xxx)>1
set(xL,'Color','k','FontWeight','bold','BackGroundColor','w',...
    'String','X (Km)')
xp1=xxx(1);yp1=yyy(1);xp2=xxx(2);yp2=yyy(2);
xc=linspace(xp1,xp2,100);
yc=linspace(yp1,yp2,100);
deltx=abs(xp1-xp2);delty=abs(yp1-yp2);PL=hypot(deltx,delty);
PROFX=linspace(0,PL,100);% profile X
gobsP=interp2(x,y,data,xc,yc); % profile value Obs grav.
gcalcP=interp2(x,y,gcalc,xc,yc);%profile calc grav.
zcalcP=interp2(x,y,zcalc,xc,yc);% profile basement
plotcross(x,y,xc,yc,PROFX,gobsP,gcalcP,zcalcP,gcalc,zcalc,wedge)
save('tempinv.mat','gobsP','gcalcP','zcalcP','xc','yc','PROFX','-append')
end
end

function plotcross(x,y,xc,yc,PROFX,TC,TCC,ZC,gcalc,zcalc,wedge) % plot of extracted profile
subplot(2,2,1)
plot(PROFX,TC,'+r',PROFX,TCC,'-k','linewidth',1);grid on;box on;
xlim([min(PROFX) max(PROFX)]);
ylabel('mGal','FontSize',10,'FontWeight','bold');
xlabel('Distance (Km)','FontSize',10,'FontWeight','bold')
legend('Observed','Calculated');
set(gca,'FontSize',10,'FontWeight','bold')
%%%%%%%%%%%%%%%
subplot(2,2,3)
dfz=max(ZC)+(max(ZC)/5);
ZC(end+1:end+3)=[dfz dfz ZC(1)]; 
PROFX(end+1:end+3)=[PROFX(end) PROFX(1) PROFX(1)];
plot(PROFX(1:end-3),-ZC(1:end-3),'-k','linewidth',2);
patch(PROFX,-ZC,[.1 .1 .1],'FaceAlpha',.8);
ylabel('Depth (Km)','FontSize',10,'FontWeight','bold')
xlabel('Click to Export Profile Data','Color','y','BackGroundColor','k')
xlim([min(PROFX) max(PROFX)]);
ylim([min(-ZC) 0]);
box on
grid on
set(gca,'FontSize',10,'FontWeight','bold')
subplot(2,2,2);mapper(x,y,gcalc,'mGal','Inverted Anomaly',2,2,wedge)
line(xc,yc,'color','k','Linewidth',2)
subplot(2,2,4);mapper(x,y,-zcalc,'Km','Calculated Depth',1,2,wedge)
line(xc,yc,'color','k','Linewidth',2)
set(gcf,'WindowButtonDownFcn',@decidesavcross)
set(findobj(gcf,'Tag','crossb'),'enable','off')
end

function decidesavcross(~,~) %%% dialog window in crossection view 
set(gcf,'outerposition',[0.255 .1 .745 .9])
str = {'Save Profile Data';'Return'};
      [s,v] = listdlg('PromptString','Proccess',...
                      'SelectionMode','single',...
                      'ListString',str,'ListSize',[150 50]);
if v==1;
    switch s
      case 1
      savcross    
      case 2
      selectionout_inv     
    end
    set(gcf,'WindowButtonDownFcn',@selectionout_inv);
end
end

function savcross(~,~) % storing profile data
[filename, pathname] = uiputfile(['profile' '.dat'], 'Set a Profile Name');
kk=[pathname filename];
if ischar(kk)
load('tempinv.mat','xc','yc','PROFX','gobsP','zcalcP','gcalcP')    
fidd=fopen(kk,'wt');
matrix=[xc' yc' PROFX' gobsP' gcalcP' -zcalcP'];
fprintf(fidd,'%s %s %s %s %s %s','Long','Lat','PROFX','Grav','CalcGrav','CalcZ');
fprintf(fidd,'\n');
for jj=1:numel(xc); % Write matrix
fprintf(fidd,'%f %f %f %f %f %f',matrix(jj,:));
fprintf(fidd,'\n');
end
fclose(fidd);
xlabel('PROFILE DATA STORED','BackGroundColor','g');
end
end

function datasout(src,~) % storing outputs (inversion/forward)
srcLb=get(src,'Label');
load('tempinv.mat');
if mapsty==0;matrix=data;ext='-gobs';extu='.grd';end
if mapsty==1;matrix=-zcalc;ext='-zcalc';extu='.grd';end
if mapsty==2;matrix=gcalc;ext='-gcalc';extu='.grd';end
if mapsty==3;matrix=data-gcalc;ext='-Gdiff';extu='.grd';end
if mapsty==11;matrix=-data;ext='-MZ';extu='.grd';wedge=0;end
if mapsty==22;matrix=gcalcf;ext='-MGrv';extu='.grd';wedge=0;end
xmin=min(min(x(wedge+1:end-wedge,wedge+1:end-wedge)));
xmax=max(max(x(wedge+1:end-wedge,wedge+1:end-wedge)));
ymin=min(min(y(wedge+1:end-wedge,wedge+1:end-wedge)));
ymax=max(max(y(wedge+1:end-wedge,wedge+1:end-wedge)));
if mapsty<4;matrix=matrix(wedge+1:end-wedge,wedge+1:end-wedge);end
if mapsty==4;matrix=[(1:numel(rmstor))' rmstor' erstor'];ext='-rmserr';extu='.dat';end
switch srcLb
    case 'Export Screen-Data'
    [filename, pathname] = uiputfile([filnam ext extu],'Set file name');
    kk=[pathname filename];
    if ischar(kk) && mapsty<4;grdout(matrix,xmin,xmax,ymin,ymax,kk);end
    if ischar(kk) && mapsty==4;save(kk,'matrix','-ascii');end
    if ischar(kk) && mapsty>5;grdout(matrix,xmin,xmax,ymin,ymax,kk);end
    case 'Export All-Data'
    [filename, pathname] = uiputfile([filnam '.gra'],'Set file name');
    kk=[pathname filename(1:end-4)];
    if ischar(kk)
    zcalc=zcalc(wedge+1:end-wedge,wedge+1:end-wedge);
    gcalc=gcalc(wedge+1:end-wedge,wedge+1:end-wedge);
    data=data(wedge+1:end-wedge,wedge+1:end-wedge);    
    grdout(-zcalc,xmin,xmax,ymin,ymax,[kk '-zcalc.grd'])
    grdout(gcalc,xmin,xmax,ymin,ymax,[kk '-gcalc.grd'])
    grdout(data-gcalc,xmin,xmax,ymin,ymax,[kk '-Gdiff.grd'])
    grdout(data,xmin,xmax,ymin,ymax,[kk '-Gobs.grd'])
    matrix=[(1:numel(rmstor))' rmstor' erstor'];
    save([kk '-rmserr.dat'],'matrix','-ascii');
    end
    case 'Save-Project'
    savproj
end
end

function savproj(~,~) %store as project file *-CHproj.mat file
clc;clear all;clear global;
[filename, pathname] = uiputfile('.mat', ' Set a Project Name');
kk=[pathname filename(1:end-4) '-CHproj.mat'];
if ischar(kk)
copyfile('tempinv.mat', kk);
end
end

function imageout(src,~) %capture screen as image 
set(gcf,'outerposition',[0.255 .1 .745 .9])
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
print(trgtf,ext, '-r300');
xlabel(['Image exported as a ...' srcLb ' -file'])
end
end

function selectionout_forw(~,~) %%% list of outputs for display (forward mode)
set(gcf,'outerposition',[0.255 .1 .745 .9])
str = {'Calculated Gravity';'Depth Model'};
      [s,v] = listdlg('PromptString','Select Map:',...
                      'SelectionMode','single',...
                      'ListString',str,'ListSize',[150 60]);
load('tempinv.mat','x','y','data','gcalcf')
if v==1;
    hax=findall(gcf,'type','axes');
    delete(hax);
  switch s
      case 1
      mapper(x,y,gcalcf,'mGal','Calculated Gravity',22,2,0)
      case 2
      mapper(x,y,-data,'Km','Depth Model',11,2,0)
  end
end
end

function selectionout_inv(~,~)%%% list of outputs for display (inv. mode)
set(gcf,'outerposition',[0.255 .1 .745 .9])
str = {'1-Calculated Depth';'2-Inverted Anomaly';'3-Anomaly Difference';
      '4a-RMS Plot';'4b-Max. Err Plot';'Observed Anomaly';'All Maps'};
      [s,v] = listdlg('PromptString','Select Output Map:',...
                      'SelectionMode','single',...
                      'ListString',str,'ListSize',[200 120]);
load('tempinv.mat','x','y','data','zcalc','gcalc','rmstor','erstor','wedge')
if v==1;
    hax=findall(gcf,'type','axes');
    delete(hax);
  switch s
      case 1
      mapper(x,y,-zcalc,'Km','Calculated Depth',1,2,wedge)
      set(findobj(gcf,'Tag','crossb'),'enable','on')
      case 2
      mapper(x,y,gcalc,'mGal','Inverted Anomaly',2,2,wedge)
      set(findobj(gcf,'Tag','crossb'),'enable','on')
      case 3
      mapper(x,y,data-gcalc,'mGal','Anomaly Difference',3,2,wedge)
      set(findobj(gcf,'Tag','crossb'),'enable','on')
      case 4
      rmsplot(rmstor)
      set(findobj(gcf,'Tag','crossb'),'enable','off')
      case 5
      ersplot(erstor)
      set(findobj(gcf,'Tag','crossb'),'enable','off')
      case 6
      mapper(x,y,data,'mGal','Observed Anomaly',0,2,wedge)
      set(findobj(gcf,'Tag','crossb'),'enable','on')
      case 7
      drawnow;plot(.5,.5,'k+');title('preparing plots...');axis off;drawnow    
      subplot(2,2,1);mapper(x,y,data,'mGal','Observed Anomaly',1111,2,wedge)    
      subplot(2,2,2);mapper(x,y,gcalc,'mGal','Inverted Anomaly',1111,2,wedge)
      subplot(2,2,3);mapper(x,y,data-gcalc,'mGal','Anomaly Difference',3,2,wedge)
      subplot(2,2,4);mapper(x,y,-zcalc,'Km','Calculated Depth',1111,2,wedge)
      set(findobj(gcf,'Tag','crossb'),'enable','off')
  end
end
end

function rmsplot(rmstor) %%% plot rms graph
      plot(rmstor,'-ko','MarkerFaceColor','r','MarkerSize',5);
      set(gca,'FontSize',10,'FontWeight','bold')
      sent1=['RMS(1)=' num2str(rmstor(1))];
      sent2=['RMS(end)=' num2str(rmstor(end))];
      title ([sent1 '    ' sent2])
      xlabel('Iteration number');ylabel('RMS');
      xlim([1 numel(rmstor)]);ylim([floor(rmstor(end)) rmstor(1)+0.1])
      grid on
      mapsty=4;
      save('tempinv.mat','mapsty','-append')
end


function ersplot(erstor) %%% plot mx. error graph
      plot(erstor,'-ko','MarkerFaceColor','b','MarkerSize',5);
      set(gca,'FontSize',10,'FontWeight','bold')
      sent1=['ERR(1)=' num2str(erstor(1))];
      sent2=['ERR(end)=' num2str(erstor(end))];
      title ([sent1 '    ' sent2])
      xlabel('Iteration number');ylabel('Max. Error');
      xlim([1 numel(erstor)]);ylim([floor(erstor(end)) ceil(erstor(1))])
      grid on
      mapsty=4;
      save('tempinv.mat','mapsty','-append')
end

function [rms,ers]=rmscalc(D1,D2,wedge) %% rms and max.error calculator
 D1=D1(wedge+1:end-wedge,wedge+1:end-wedge);
 D2=D2(wedge+1:end-wedge,wedge+1:end-wedge);
 dg2=(D1-D2).^2;
 rms=sqrt(sum(sum(dg2))./((numel(D1))));
 ers=max(max(abs(D1-D2)));
 end
 
function instaRMSplot(step,rmstor,erstor,termin,maxiter,lmt)
drawnow %% instant plot of rms or mx. error during on-going iteration
if termin==0;
plot(1:step,rmstor(1:step),'-k+');grid on;
xlim([1 maxiter]);ylim([0 rmstor(1)]);
title(['RMS1=' num2str(rmstor(1)) '  RMSend=' num2str(rmstor(step))],'Color','r')
if lmt==0;xlim([1 step]);
title(['RMS1=' num2str(rmstor(1)) '  RMSend=' num2str(rmstor(step))],'Color','b')
xlabel(['performed iteration=' num2str(step)])
end
end

if termin==1;
plot(1:step,erstor(1:step),'-k+');grid on;
xlim([1 maxiter]);ylim([0 erstor(1)]);
title(['ERR1=' num2str(erstor(1)) '  ERRend=' num2str(erstor(step))],'Color','r')
if lmt==0;xlim([1 step]);
title(['ERR1=' num2str(erstor(1)) '  ERRend=' num2str(erstor(step))],'Color','b')
xlabel(['performed iteration=' num2str(step)])
end
end
set(gca,'YTickLabel',[]);
drawnow
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function termino(~,event) 
%%% retrieve iteration stop criterion settings, modify related GUI items
t=get(findobj(gcf,'Tag','tabl2'),'Data');
sc=event.Indices;
if numel(sc)==2
    scc=sc(end);
if scc==2;
tab3=findobj(gcf,'Tag','tabl3');delete(tab3);    
set(findobj(gcf,'Tag','tabl2'),'Data',{false,true, cell2mat(t(3))})
if exist('tempinv.mat','file')==0;save('tempinv.mat');end
C = who('-file','tempinv.mat');
    if ismember('criterio', C);
    load('tempinv.mat','criterio');
    else
    criterio=0.01;
    end
uitable(gcf,'units','normalized','Position',[.01,.53,.98,.065],...
'ColumnName',{'current value              '},'Tag','tabl3',...
'Rowname',{'set min.Rms'},...
'Data',criterio,'ColumnEditable',[true],'CellEditCallback',@setcriterio);
set(findobj(gcf,'Tag','startbut'),'Position',[.01,.48,.98,.04],...
    'enable','on')
end

if scc==1;
tab3=findobj(gcf,'Tag','tabl3');delete(tab3);    
set(findobj(gcf,'Tag','tabl2'),'Data',{true,false, cell2mat(t(3))})
if exist('tempinv.mat','file')==0;save('tempinv.mat');end
C = who('-file','tempinv.mat');
    if ismember('criterio', C);
    load('tempinv.mat','criterio');
    else
    criterio=0.05;
    end
uitable(gcf,'units','normalized','Position',[.01,.53,.98,.065],...
'ColumnName',{'current value              '},'Tag','tabl3',...
'Rowname',{'set Max.error'},...
'Data',criterio,'ColumnEditable',[true],'CellEditCallback',@setcriterio);
set(findobj(gcf,'Tag','startbut'),'Position',[.01,.48,.98,.04],...
    'enable','on')
end
end
end


function setcriterio(~,~) %%% setting threshold value of rms or err 
criterio=get(findobj(gcf,'Tag','tabl3'),'Data');
save('tempinv.mat','criterio','-append')
end

%%%%%%%%%%%%%%%%%%%%%%%%% GUI settings
function list1w(T,nx,ny,xmin,xmax,ymin,ymax,dx,dy,k) %%% print mesh info of input grid
set(findobj(gcf,'Tag','listb1'),'value',1);
s0='Grid Info:';
s1=['Source : ' k];
s2=['NX : ' num2str(nx) '   NY : ' num2str(ny)];
s3=['Grid Spacing     dx: ' num2str(dx) '   dy: ' num2str(dy)];
s4=['Xmin - Xmax:   ' num2str(xmin) '  /   ' num2str(xmax)];
s5=['Ymin - Ymax:   ' num2str(ymin) '  /   ' num2str(ymax)];
s6=['Zmin - Zmax:   ' num2str(min(min(T))) '  /  ' num2str(max(max(T)))];
str={s0;s1;s2;s3;s4;s5;s6};
set(findobj(gcf,'Tag','listb1'),'string',str);
end

function posfig1(~,~)  % reset position of GUI
set(gcf,'outerposition',[0 .1 .25 .9])
end

function [matrix,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy,err]=grid2checker(k);
fidc=fopen(k);
header= fread(fidc,4,'*char' )';
fclose(fidc);
c1=strcmp(header,'DSAA');
c2=strcmp(header,'DSRB');
sumc=sum([c1 c2]);
if sumc>0
switch c1
    case 1
[matrix,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy]=lodgrd6txt(k); %format surfer6 text
    case 0
[matrix,x,y,nx,ny,xmin,xmax,ymin,ymax,dx,dy]=lodgrd7bin(k); %surfer7 binary       
end
%%%%%%%%%%%%%check dimension even or odd numbered..if odd 
%%%%%%%%%%%%%initialise to even size by cutting one column or one row at east or north
if mod(nx,2)==1;nx=nx-1;matrix(:,end)=[];x(:,end)=[];y(:,end)=[];xmax=max(max(x));end
if mod(ny,2)==1;ny=ny-1;matrix(end,:)=[];y(end,:)=[];x(end,:)=[];ymax=max(max(y));end
err=0;
if dx~=dy;err=2;end
if any(isnan(matrix(:)));err=4;end
else
err=3;matrix=0;x=0;y=0;nx=0;ny=0;xmin=0;xmax=0;ymin=0;ymax=0;dx=0;dy=0;    
end
end


function mapper(x,y,matrix,unitt,tit,mapsty,vie,wedge) %%map viewer
xx=x(wedge+1:end-wedge,wedge+1:end-wedge);
yy=y(wedge+1:end-wedge,wedge+1:end-wedge);
matrixx=matrix(wedge+1:end-wedge,wedge+1:end-wedge);
if vie==2
contourf(xx,yy,matrixx,30);shading flat;
rotate3d off;
axis equal
axis tight
end
if vie==3
surf(xx,yy,matrixx,'FaceColor','interp','EdgeColor','none','FaceLighting','gouraud')
rotate3d on
box on
end
if vie==4
pcolor(xx,yy,matrixx)
rotate3d off
axis equal
axis tight
end

set(gca,'FontSize',10,'FontWeight','bold')
h=colorbar('eastoutside');title(h,unitt,'FontWeight','bold');
set(h,'FontSize',10,'FontWeight','bold')
xlabel('X (Km)');ylabel('Y (Km)');title(tit)
xlim([min(min(x)) max(max(x))])
ylim([min(min(y)) max(max(y))])
%if mapsty==3;caxis([min(min(matrix)) max(max(matrix))]);end
save('tempinv.mat','mapsty','-append')
end

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

function densmodelfit(~,~)
clc;%%%%close figures
closwindows(0)
%%%%%%%new figure
fig2=create_actwindow;set(fig2,'name','Model')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=uimenu('Parent',fig2,'Label','Import Data','ForeGroundColor','b');
uimenu('Parent',m1,'Label','Load Depth- Density Contrast data (*.txt, *.dat)','CallBack',@loaddens)
text(.01,.4,[{'Exponential fit tool for calculating r0 and lambda parameters';...
'----------------------------------';'';...
'1-Import two column data from an ascii file [depth vs dens.contrast]';...
'note: depth (km) in negative';...
'      contrast (gr/cc) in negative'
'2-click on graph for saving outputs'}]);
axis off
end

function loaddens(src,~)
drawnow
cla;
drawnow
[filename, pathname] = uigetfile({'*.dat';'*.txt'}, 'Import data');
k=[pathname filename];
if ischar(k)
depdensm=load(k);
if exist('tempinv.mat','file')==0;save('tempinv.mat');end
set(src,'Label',['change data: ' k])
startdensfitter(depdensm)
set(gcf,'WindowButtonDownFcn',@savfitcur)
end
end

function startdensfitter(depdensm)
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
contrvec=-contrast;
depvec=-dep;
fitcurvec=fitcur;
save('tempinv.mat','fitcurvec','contrvec','depvec','r0','lambda','-append')
end



function savfitcur(~,~)
set(gcf,'outerposition',[0.255 .1 .745 .9])
str = {'Save as image...';'Export Data'};
      [s,v] = listdlg('PromptString','Proccess',...
                      'SelectionMode','single',...
                      'ListString',str,'ListSize',[150 50]);
if v==1;
    switch s
      case 1
      ext='-dpng';
      [filename, pathname] = uiputfile('*.png','Set a filename');
      kk=[pathname filename];
      if ischar(kk)
      drawnow
      print(kk,ext, '-r300');
      xlabel('Exported as a ...image -file')
      end
      case 2
      load('tempinv.mat','fitcurvec','contrvec','depvec','r0','lambda')
      [filename, pathname] = uiputfile('*.dat','Set a filename');
      kk=[pathname filename];
      if ischar(kk)
      matrix=[depvec contrvec fitcurvec];    
      save(kk,'matrix','-ascii')
      end
      
    end
end
end
