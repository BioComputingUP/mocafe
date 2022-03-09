Search.setIndex({docnames:["bib","code_documentation","demo_doc/angiogenesis_2d","demo_doc/angiogenesis_3d","demo_doc/index","demo_doc/introduction_to_fenics","demo_doc/multiple_pc_simulations","demo_doc/prostate_cancer2d","demo_doc/prostate_cancer3d","index","installation","sub_code_doc/angie","sub_code_doc/angie_sub_code_doc/af_sourcing","sub_code_doc/angie_sub_code_doc/base_classes","sub_code_doc/angie_sub_code_doc/forms","sub_code_doc/angie_sub_code_doc/tipcells","sub_code_doc/expressions","sub_code_doc/fenut","sub_code_doc/fenut_sub_code_doc/fenut","sub_code_doc/fenut_sub_code_doc/log","sub_code_doc/fenut_sub_code_doc/mansimdata","sub_code_doc/fenut_sub_code_doc/parameters","sub_code_doc/fenut_sub_code_doc/solvers","sub_code_doc/litforms","sub_code_doc/litforms_sub_code_doc/prostate_cancer","sub_code_doc/litforms_sub_code_doc/xu16","sub_code_doc/math"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["bib.rst","code_documentation.rst","demo_doc/angiogenesis_2d.rst","demo_doc/angiogenesis_3d.rst","demo_doc/index.rst","demo_doc/introduction_to_fenics.rst","demo_doc/multiple_pc_simulations.rst","demo_doc/prostate_cancer2d.rst","demo_doc/prostate_cancer3d.rst","index.rst","installation.rst","sub_code_doc/angie.rst","sub_code_doc/angie_sub_code_doc/af_sourcing.rst","sub_code_doc/angie_sub_code_doc/base_classes.rst","sub_code_doc/angie_sub_code_doc/forms.rst","sub_code_doc/angie_sub_code_doc/tipcells.rst","sub_code_doc/expressions.rst","sub_code_doc/fenut.rst","sub_code_doc/fenut_sub_code_doc/fenut.rst","sub_code_doc/fenut_sub_code_doc/log.rst","sub_code_doc/fenut_sub_code_doc/mansimdata.rst","sub_code_doc/fenut_sub_code_doc/parameters.rst","sub_code_doc/fenut_sub_code_doc/solvers.rst","sub_code_doc/litforms.rst","sub_code_doc/litforms_sub_code_doc/prostate_cancer.rst","sub_code_doc/litforms_sub_code_doc/xu16.rst","sub_code_doc/math.rst"],objects:{"mocafe.angie":[[12,0,0,"-","af_sourcing"],[13,0,0,"-","base_classes"],[14,0,0,"-","forms"],[15,0,0,"-","tipcells"]],"mocafe.angie.af_sourcing":[[12,1,1,"","ClockChecker"],[12,1,1,"","ConstantSourcesField"],[12,1,1,"","RandomSourceMap"],[12,1,1,"","SourceCell"],[12,1,1,"","SourceMap"],[12,1,1,"","SourcesManager"],[12,3,1,"","sources_in_circle_points"]],"mocafe.angie.af_sourcing.ClockChecker":[[12,2,1,"","clock_check"]],"mocafe.angie.af_sourcing.SourceMap":[[12,2,1,"","get_global_source_cells"],[12,2,1,"","get_local_source_cells"],[12,2,1,"","remove_global_source"]],"mocafe.angie.af_sourcing.SourcesManager":[[12,2,1,"","apply_sources"],[12,2,1,"","remove_sources_near_vessels"]],"mocafe.angie.base_classes":[[13,3,1,"","fibonacci_sphere"]],"mocafe.angie.forms":[[14,3,1,"","angiogenesis_form"],[14,3,1,"","angiogenic_factor_form"],[14,3,1,"","cahn_hillard_form"],[14,3,1,"","vascular_proliferation_form"]],"mocafe.angie.tipcells":[[15,1,1,"","TipCell"],[15,1,1,"","TipCellManager"],[15,1,1,"","TipCellsField"]],"mocafe.angie.tipcells.TipCell":[[15,2,1,"","get_radius"],[15,2,1,"","is_point_inside"],[15,2,1,"","move"]],"mocafe.angie.tipcells.TipCellManager":[[15,2,1,"","activate_tip_cell"],[15,2,1,"","compute_tip_cell_velocity"],[15,2,1,"","get_global_tip_cells_list"],[15,2,1,"","move_tip_cells"],[15,2,1,"","revert_tip_cells"]],"mocafe.angie.tipcells.TipCellsField":[[15,2,1,"","add_tip_cell"],[15,2,1,"","compute_phi_c"],[15,2,1,"","eval"]],"mocafe.expressions":[[16,1,1,"","EllipseField"],[16,1,1,"","EllipsoidField"],[16,1,1,"","PythonFunctionField"],[16,1,1,"","SmoothCircle"],[16,1,1,"","SmoothCircularTumor"]],"mocafe.fenut":[[18,0,0,"-","fenut"],[19,0,0,"-","log"],[20,0,0,"-","mansimdata"],[21,0,0,"-","parameters"],[22,0,0,"-","solvers"]],"mocafe.fenut.fenut":[[18,3,1,"","build_local_box"],[18,3,1,"","divide_in_chunks"],[18,3,1,"","flatten_list_of_lists"],[18,3,1,"","get_mixed_function_space"],[18,3,1,"","is_in_local_box"],[18,3,1,"","is_point_inside_mesh"],[18,3,1,"","load_parameters"],[18,3,1,"","setup_pvd_files"],[18,3,1,"","setup_xdmf_files"]],"mocafe.fenut.log":[[19,1,1,"","DebugAdapter"],[19,1,1,"","InfoCsvAdapter"],[19,3,1,"","confgure_root_logger_with_standard_settings"]],"mocafe.fenut.log.DebugAdapter":[[19,2,1,"","process"]],"mocafe.fenut.log.InfoCsvAdapter":[[19,2,1,"","process"]],"mocafe.fenut.mansimdata":[[20,3,1,"","save_sim_info"],[20,3,1,"","setup_data_folder"]],"mocafe.fenut.parameters":[[21,1,1,"","Parameters"],[21,3,1,"","from_dict"],[21,3,1,"","from_ods_sheet"]],"mocafe.fenut.parameters.Parameters":[[21,2,1,"","as_dataframe"],[21,2,1,"","get_value"],[21,2,1,"","is_parameter"],[21,2,1,"","is_value_present"],[21,2,1,"","set_value"]],"mocafe.fenut.solvers":[[22,1,1,"","PETScNewtonSolver"],[22,1,1,"","PETScProblem"]],"mocafe.fenut.solvers.PETScNewtonSolver":[[22,2,1,"","solver_setup"]],"mocafe.fenut.solvers.PETScProblem":[[22,2,1,"","F"],[22,2,1,"","J"]],"mocafe.litforms":[[24,0,0,"-","prostate_cancer"],[25,0,0,"-","xu16"]],"mocafe.litforms.prostate_cancer":[[24,3,1,"","df_dphi"],[24,3,1,"","prostate_cancer_chem_potential"],[24,3,1,"","prostate_cancer_form"],[24,3,1,"","prostate_cancer_nutrient_form"]],"mocafe.litforms.xu16":[[25,3,1,"","xu2016_nutrient_form"],[25,3,1,"","xu_2016_cancer_form"]],"mocafe.math":[[26,3,1,"","estimate_cancer_area"],[26,3,1,"","estimate_capillaries_area"],[26,3,1,"","shf"],[26,3,1,"","sigmoid"]],mocafe:[[16,0,0,"-","expressions"],[26,0,0,"-","math"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[0,2,3,5,6,7,8,10,14,15,16,25,26],"000":[2,3,5,6,7,8],"0000":[6,20],"0001":[6,20],"001":[5,6,7,8],"0019989":0,"009":0,"01":[6,7,8],"0149422":0,"03":0,"0e6":[6,7,8],"1":[0,2,3,5,6,7,8,13,14,15,16,18,25,26],"10":[0,10],"100":[0,2,3,6,7,8,16,26],"1000":[6,7,8],"1003":[6,7,8],"1007":0,"1016":0,"1058693490":0,"1073":0,"11":[0,2,3,11,12,14,15],"1145":0,"11588":0,"130":[6,7,8],"1371":0,"14":[0,1,5],"15":[0,5,9],"150":[6,7,8],"16":[0,6,7,8,24],"1615791113":0,"17":[0,7],"2":[0,2,3,5,6,7,8,10,14,15,18],"200":3,"2000":[7,8],"2011":[0,2,3,14,15],"2013":0,"2014":0,"2015":0,"2016":[0,7,8],"2017":0,"2019":[10,18],"2021":0,"20553":0,"225":[],"2368":[],"239":[],"25":5,"2566630":0,"27s_laws_of_diffus":0,"2d":[2,3,5,6,7,8,15],"3":[0,2,3,10,15,18],"30":5,"300":[2,6],"319":0,"32":5,"33287":[],"365":[6,7],"37":2,"375":2,"3d":[4,5,12,14,15],"4":[2,3,6,7,8,10,14,15,18],"400":6,"4th":14,"5":[0,2,5,6,7,8,18,25],"500":8,"512":[],"515":0,"52461":0,"52462":0,"548":0,"6":[0,18,25],"600":[6,7,8],"642":[],"6_9":[],"6e5":[6,7,8],"7":[0,18],"73":[6,7,8],"75":[6,7,8],"8":[3,18],"9":0,"961":[6,7,8],"978":0,"\u00e1":0,"\u00e9":0,"\u00f8lgaard":0,"aln\u00e6":0,"boolean":2,"case":[2,3,5,7,8,10],"class":[2,5,7,12,13,15,16,17,18,19,21,22],"default":[2,5,7,10,11,18,20,26],"do":[2,3,5,6,7,8,10,15],"final":[2,5,7],"float":[16,18,20,26],"function":[1,2,5,6,7,8,12,14,15,16,18,24,25,26],"hern\u00e1ndez":[],"import":[2,3,5,6,7,8,10],"int":[2,3,12,15,18,22],"long":18,"new":[2,5,6,7,8,15,20,21],"poir\u00e9":[],"public":[24,25],"return":[2,12,13,14,15,18,19,20,21,24,25,26],"short":[2,3,7,8],"true":[2,3,6,8,12,15,18,20,21],"try":10,"var":10,"while":[2,3,7],A:[0,2,4,6,7,8,12,18,24],And:[2,5,6,7,8,10],As:[2,3,5,6,7,8],At:6,By:11,For:[2,3,5,6,7,12,14,15,20,24,25],If:[5,7,10,12,14,15,18,20],In:[1,2,3,5,6,7,8,9,10,14,15,18,26],It:[5,12,21,24],ONE:0,Of:[3,7,8,10],One:[2,3,5,8],The:[0,2,3,6,7,8,9,10,12,14,15,16,18,19,20,21,24,25],Then:[2,3,5,6,7,8,10],There:10,These:[2,7],To:[2,3,5,6,7,10],Will:20,With:[2,3,7,8],_:[],__file__:[2,3,6,7,8],__name__:[3,6,7,8],a_valu:6,abh:[0,5,9],abl:[3,5,7,8],about:[5,21],abov:[2,3,5,6,7,8,10,15],academi:0,access:[0,5,12,23],accord:[2,5,7,15,25],accordingli:[2,3,6,7,8],account:2,acm:0,across:[2,5],act:[],action:6,activ:[2,3,15],activate_tip_cel:[2,3,15],actual:[2,3,5,7],ad:[2,3,14,20],adapt:19,add:[10,15],add_tip_cel:15,addit:2,addition:2,additionali:7,adimdiment:[6,7,8],adimension:[6,7,8],adiment:[6,7,8],advanc:5,advantag:7,af:[2,3,12,14,15],af_0:[2,3,14],af_:2,af_at_point:15,af_c:2,af_expression_funct:[],af_p:[2,14,15],af_sourc:[1,2,3,11],afexpressionfunct:[],after:[2,3,6,7],again:[2,3,5,8],agent:[11,12],agreement:7,aim:[],al:[2,3,7,12,14,15,25],algebra:[5,6,7,22],algorithm:[2,5,11,13],all:[2,3,4,5,7,10,11,15,24],allow:[],almost:18,alnaeslolgaard:[0,1,5],along:[2,7],alpha:14,alpha_p:[2,14,15],alpha_t:[2,14],alreadi:[2,3,7,10],also:[2,3,4,5,7,9,12,15,24,25],altern:[],alwai:[3,8],amg:3,among:[],an:[0,2,3,5,7,8,9,11,12,16,20,21],analyz:7,ander:0,angi:[1,2,3,9],angiogen:[2,12,14,15],angiogenesi:[0,1,4,9,11,12,14,15],angiogenesis_2d:2,angiogenesis_2d_initial_condit:[],angiogenesis_3d:3,angiogenesis_form:[2,3,14],angiogenic_factor_form:[2,3,14],ani:[2,3,5,7,8,10,12,21],anim:5,anoth:2,anymor:10,apoptosi:[7,24],append:[],appli:[0,2,3,7,12,15],applic:0,apply_sourc:[2,3,12],appreci:[],approach:11,appropri:[3,8],approxim:[2,5],apt:9,ar:[2,3,5,6,7,8,11,12,14,15,18,20,21],archiv:0,area:26,arg0:22,arg1:22,arg2:22,arg3:22,arg:19,argument:[2,5,6,7,14,19,20,24,25],argv:7,around:[],arrai:[6,7,8],articl:0,as_datafram:21,ask:[2,20],asm:[],aspect:17,assign:[2,3,5,6,7,8],associ:24,assum:7,august:0,author:7,auto:20,auto_enumer:[2,3,6,7,8,20],automat:[2,6,7,10,20],autoremov:10,auxiliari:[2,3,14],auxilliari:2,avail:10,averag:7,avoid:7,b:[0,2,3,6,7,8,18,26],backend:[5,22],backward:[5,24,25],ban:[],bar:[2,3,6,7,8],base:[7,9,13,25],base_class:[1,11],base_loc:[],basecel:12,bash_complet:10,basi:[14,16],basic:21,bc:22,becaus:[2,5,7,8],becom:2,been:[3,10],befor:[2,3,5,7],begin:[5,7],behav:2,behaviour:[2,5],being:[],below:[1,2,5,10,14,18],bengzon:[],benjamin:0,berlin:[],best:[5,7,8,18],better:[7,9],between:[7,26],beyond:[],bibliographi:9,bigger:18,bin:10,bind:10,bit:[2,7],black:5,blechta:0,block:[],blood:[3,12],book:5,bool:20,border:[16,18],border_width:18,bot:[],both:[2,3,5,7,8,19],box:[5,10,18],boxmesh:[3,8],brief:4,briefli:[2,5,14],bug:10,build:[10,18,24,25],build_local_box:18,built:[5,7,10],builtin:5,c0:14,c:[2,3,5,7,12,14,15,25,26],c_0:[2,3],c_0_exp:2,c_c:2,c_exp:3,c_x:5,c_y:5,cach:10,cahn:[2,14],cahn_hillard_form:14,calcul:[],call:[2,5,6,7,14,15,19,20,22],came:5,can:[1,2,3,5,6,7,8,9,10,14,15,16,18,19,20],cancer:[0,1,2,4,6,9,23,24,25,26],cannot:2,canon:5,capillari:[2,7,14,15,25,26],capilllari:2,care:[2,3,5,8,12],carlo:0,castro:0,caus:2,cdot:[2,5,6,7,14,15,26],cell:[2,3,7,12,15],cell_radiu:12,center:[2,5,6,7,8,12,15,16,26],central:26,certain:2,cg1_element:7,cg:[2,3,6,7,8,18],cgi:0,chang:[2,3,6,7,8,9,10],chapter:[],check:[2,3,5,12,15,18,21],checker:12,chem_potenti:14,chem_potential_const:24,chemic:[5,24],chempot_const:[6,7,8,24],chi:[2,6,7,8,15,24],chi_valu:6,choic:[3,5,8,18],choos:[5,7,8],chose:7,chosen:7,chri:0,chunk:18,circl:[2,3,5,12,15,16],circle_radiu:12,circular:16,cite:[2,5,24,25],clarif:2,clarifi:7,clean:10,clear:[],clearli:5,click:[2,3,4,5,6,7,8],clock:12,clock_check:12,clockcheck:12,close:5,closer:2,closest_tip_cel:15,cma:0,coarsen:0,code:[4,9,15,20],coher:15,collabor:[3,7,8,11,24,25],collaps:[2,3,6,7,8],collect:5,column:21,com:10,come:2,comm:[2,3,6,7,8,18,22],comm_world:[2,3,6,7,8,18],command:[2,7,10,20],comment:7,common:[7,10,14,24],commun:[10,18],compar:3,compil:10,complet:[1,2,5,6,7,9,24,25],complex:2,compon:7,compos:[2,7,18],compromis:[],compulsori:21,comput:[0,3,5,7,8,10,15,18],compute_phi_c:15,compute_tip_cell_veloc:15,conacyt:0,conatin:[],concentr:[2,5,7,14,15],condit:[6,12,14,15,25],conductor:5,confgure_root_logger_with_standard_set:19,confid:[],configur:[7,8,19],conserv:18,consid:[2,3,5,7,8,10,14],constant:[2,3,5,7,14,15,24,25],constant_valu:[],constantafexpressionfunct:[],constantli:[2,7],constantsourcesfield:12,constrain:[3,8],construct:[5,7],constructor:2,consum:[7,25],consumpt:[2,14],contain:[2,6,7,9,11,12,15,17,18,20,21,23,24,25],content:6,contextu:19,continu:[2,7],continuo:18,contributor:0,conveneinc:14,conveni:[2,6,7],coordin:2,copyright:10,core:[7,21],correctli:10,correspond:[14,21,24,25],correspong:24,corvera:0,costant:14,could:[2,14],count:2,coupl:[0,7],cours:[3,5,7,8,10],cpp:[12,14,15,18,22,24],cpu:[2,3,6,7,8],creat:[2,3,5,6,7,8,15,20,21],creation_step:[12,15],credit:10,critic:5,csv:19,cube:5,current:[2,5,6,7,8,10,12,15,18,20],current_step:[6,7,8,15],cylinder_radiu:3,cylindr:3,d:[0,2,3,4,5,7,10,14],d_:[2,25],d_sigma:25,data:[1,2,3,4,5,8,10,17,20],data_fold:[2,3,6,7,8,18,19,20],datafram:21,date:20,dateandtim:20,dc:14,de:0,deactiv:[2,15],debug:19,debugadapt:19,decai:[7,24],decemb:0,decid:[5,20],decreas:2,def:[6,10],defin:[1,2,3,5,6,7,8,14,15,18,19,22,24,26],definit:[6,10,14],defint:2,degre:[2,3,5,6,7,8,18],deleg:[],deliber:5,delta:[6,7,8,24],demo:[2,3,5,6,7,8,9,12,14,15],demo_doc:[],demo_doc_jupyt:4,demo_doc_python:4,demo_in:[2,3],demo_out:[2,3,5,6,7,8],depend:[2,5,7,10],deprec:18,deriv:[2,5,7,14,24,26],desc:[2,3],describ:[2,3,5,7,8,24,25],descript:[24,25],desid:18,design:3,desktop:[3,8],detail:[5,14],determin:16,develop:[5,7,9],df:[7,14],df_dphi:24,dict:[20,21,22,25],dictionari:[7,21],did:[2,3,7,8],differ:[2,3,5,6,7,8,15,17,18],differenti:[0,3,5,7,8],difficult:[],diffus:[0,2,7,14,24,25],dimens:[3,7,8],dimenson:[],directli:22,directori:20,discours:[],discret:[6,12,14,24,25],discuss:[2,5,7,14],distanc:[2,15],distant:2,distibut:[],distribut:[2,5,7,12,13,14,18],divid:[2,18],divide_in_chunk:18,divis:18,docker:[],document:[5,7,10,24],doe:[],doesn:[],doi:0,dokken:5,dolfin:[12,14,15,18,22,24,25,26],domain:[0,2,5,7,8,12,15],don:[2,3,6,8,10],done:[7,8],dot:5,doubl:[6,7,8,24,25],down:7,download:[2,3,4,5,6,7,8,10],driven:[2,7],drop:2,dt:[2,3,5,6,7,8,14,24,25],duplic:7,dure:[2,3,6,7,8,20],dx:5,dynam:2,e0149422:0,e19989:0,e:[0,2,5,7,12,15,26],each:[2,5,6,7,12,14,15,18,20],eas:[3,8],easi:[2,6,7],easier:[2,6,10],easiest:7,easili:[2,7],easli:[2,3,6,7,8],east:12,effect:2,effici:[3,5,7],effort:[3,8],egg:10,either:19,elabor:18,element:[2,5,7,8,14,18],element_typ:18,elimin:[],ellips:[8,16],ellipsefield:[6,7,16],ellipsoid:[8,16],ellipsoidfield:[8,16],ellipt:7,elment:5,els:[2,3,6,7,8,15],empti:7,en:0,encourag:5,encyclopedia:0,end:6,endotheli:2,energet:14,engin:0,enough:[],ensur:5,enter:[],entir:[2,5,7],epsilon:[2,6,7,8,14,24],equal:[2,12,18,20,24],equal_for_all_p:[],equat:[0,3,7,8,14,24,25],error:[2,3,6,7,20],error_msg:20,es:0,especi:20,espress:[],estim:[7,26],estimate_cancer_area:26,estimate_capillaries_area:26,et:[2,3,7,12,14,15,25],etc:[2,10,17],eugenia:0,euler:[5,24,25],ev:13,eval:15,evalu:15,even:[3,7,8,18],everi:[5,7,15],everyth:[2,3,7,10,21],everywher:2,evolut:[5,7,25],exactli:[2,3,7,8],exampl:[4,5,12,14,15,18],excel:[5,7],except:[2,3],execut:[6,10,20],execution_tim:[6,20],exist:[2,3],expect:2,experi:7,experienc:[5,7],explain:[2,5,10],exploit:[2,3,6,7,8],express:[1,2,3,5,6,7,8,9,12,15],extens:[2,5,7],extra:19,extrem:[2,3,5],ey:[],f:[2,3,6,7,8,14,22],fact:5,factor:[2,12,14,15],fals:[2,3,7,8,12,15,18,21],familiar:[],far:2,faster:[],fct:0,featur:[5,7],feb:0,feel:5,fem:[2,5,7],fenic:[0,1,2,3,4,6,7,8,9,11,12,14,15,18,24,25],fenicsproject:[],fenicsx:[5,10],fenut:[1,2,3,6,7,8,9,12,14,15,24,25],few:[5,7],ffor:19,fibonacci:13,fibonacci_spher:13,fick:[0,2,7],field:[0,1,2,3,4,6,7,8,9,11,12,14,15,16,20,23,24,25,26],figur:5,file:[2,3,5,6,7,8,10,17,18,19,20,21],file_af:[2,3],file_c:[2,3],file_fold:[2,3,6,7,8],file_nam:[2,3,18],find:[1,2,3,5,7,8,9,10,14],finger:6,finit:[2,5,8,14],finiteel:7,fir:[],first:[2,3,5,6,7,10,20],flat:18,flatten:18,flatten_list_of_list:18,flow:5,fluid:5,fo:24,folder:[2,3,6,7,8,10,18,19,20],folder_nam:20,folder_path:[2,3,6,7,8,20],follow:[1,2,3,5,6,7,8,9,10,14,15,20],follw:14,form:[0,1,3,6,8,11,21,24,25],form_af:[2,3],form_ang:[2,3],format:7,formul:0,formula:[5,15],forum:7,found:15,fpradelli94:10,frac:[2,5,7,14,15,26],frame:21,fredrik:[],free:[0,5],frequenc:0,from:[2,3,5,6,7,8,10,12,19,21,25],from_dict:[6,7,8,21],from_ods_sheet:[2,3,21],fu:[2,3],full:[1,12],fun:[2,3],function_spac:[2,3,6,7,8],functionspac:[5,7,26],further:14,futur:10,g:[0,2,5,6,7,8,15],g_m:[2,15],galerk:7,galleri:[2,3,5,6,7,8,9],gamg:[3,6,7,8],gamma:[5,6,7,8,24],garth:0,ge:2,gear:5,gener:[2,3,4,5,6,7,8,12,14,18,20,25],genericmatrix:22,genericvector:22,get:[2,3,5,6,10,12,15,21],get_dim:[],get_global_mesh:[],get_global_source_cel:12,get_global_tip_cells_list:15,get_latest_tip_cell_funct:[2,3],get_local_mesh:[],get_local_source_cel:12,get_mixed_function_spac:[2,3,6,7,8,18],get_point_value_at_source_cel:[],get_radiu:15,get_rank:[2,3,6,7,8],get_valu:[2,3,6,7,8,21],gif:[],git:10,github:10,give:5,given:[2,3,5,6,7,8,12,15,16,18,20,21,26],given_list:18,given_valu:2,glaerkin:18,global:[12,15],gmre:[3,6,7,8],go:[5,6,7],goe:7,gomez:0,good:5,grad:[2,3,5],grad_af:[2,3,15],grad_af_function_spac:[2,3],grad_t:[2,3],gradient:[2,14,15],gradual:5,gram:[6,7,8],great:[3,8],greatest:5,group:[],grow:2,growth:[0,2,7],guarante:8,guillermo:0,gulbenkian:0,h5:6,h:[0,14,25],ha:[3,5,10,12,15,16],hake:0,han:0,hand:2,handl:[2,14],happen:2,hard:7,have:[2,3,5,6,7,8,10,20,21,26],healthi:25,heavisid:14,heavysid:26,hector:0,heidelberg:0,help:10,here:[2,3,5,6,7,8,10],hern:0,hi:3,hierarch:0,high:[2,5],higher:[2,3,8],hillard:[2,14],home:10,host:[2,10],how:[4,5],howev:[2,3,5,6,7,8,14],hpc:[3,8],html:[6,20],http:[0,10],hugh:0,hurri:[],hybrid:11,hypothesi:7,hypox:[2,12,25],i:[2,7,10,12,15],i_v_w:2,id:0,ident:[3,7,8],identifi:[18,20],ignor:[],imag:10,implement:[5,9,14,15],implment:4,imput:[],includ:[2,7,9],inde:[2,3,5,7,8,14,20],index:[0,9],induc:[2,12],inf:26,info:[17,19,20],infocsvadapt:19,inform:[2,7,10,19,21],inhibit:2,init:[3,6,7,8],init_tim:6,initi:[6,14,24,25],initial_vessel_radiu:3,initial_vessel_width:[2,3],input:[2,5,20],input_mesh:[],insert:19,insid:[2,3,5,6,7,10,15,16,18,21],inside_valu:[6,7,8,16],inspect:[7,9],instal:[2,3,6,7,8,9],instanc:[2,6,20,21],instead:[3,7,8,18],instruct:10,int_:5,integr:9,interact:2,interest:[2,5,6,7,18],interfac:[0,2,3,5,7,25],intern:[15,19,22],interpol:[2,3,6,7,8],intracomm:[18,22],introduc:2,introduct:4,introduction_to_fen:5,invit:5,involv:[2,14],io:[],ipynb:[2,3,5,6,7,8],ipython:10,is_in_ellipse_cpp_cod:[],is_in_local_box:18,is_inside_global_mesh:[],is_inside_local_mesh:[],is_paramet:21,is_point_insid:15,is_point_inside_mesh:18,is_value_pres:21,isbn:0,isciii:0,iter:[2,3,5,7,8],its:[2,7,10,14,15,19,24],itself:[2,7,14,19],j:[0,2,3,5,6,7,8,22],j_mat:[3,6,7,8],jacobian:[2,7],jan:0,jessica:0,jiangp:0,job:[2,7],johan:0,johann:0,johansson:0,jour:[],journal:0,json:18,juan:0,jun:0,jupyt:[2,3,4,5,6,7,8],just:[2,3,4,5,6,7,8,10,12,20],k:0,keep:[2,12],kehlet:0,kei:[2,7],kevin:0,keyword:19,kind:[5,14],know:5,known:7,kristian:0,krylov:[],ksp_type:[3,6,7,8],kwarg:[12,19],l:[5,6],la:22,lagrang:[5,14],lambda:[2,3,6,7,8,12,14,24],lambda_:25,lambda_phi:25,langtangen:0,langtangenlogg2017:5,languag:[0,1,5],laptop:[3,8],larg:7,larson:[],last:[2,5],later:[],latest:[],latter:[2,14],law:0,lb13:[],le:[2,14,15],lead:[2,5,6,14],least:21,leav:7,left:[2,7],legaci:[],lei:0,length:[],let:[2,5,7],level:[2,3,5,19],leverag:[5,7],libexec:10,librari:7,licens:10,like:[4,5,6,7,10],limit:2,line:[2,5,7,20],linear:[2,7,22],linearli:2,link:[2,3,6,7,8],linux:10,list:[6,7,10,12,15,18,24],list_of_list:18,liter:[6,7,8],litform:[1,6,7,8,9],liu:0,ll17:[0,5],ll:[3,8,19],lmbda:14,load:[3,21],load_paramet:18,load_random_st:[],loading_messag:6,local:[2,10,12,18],local_box:18,local_mesh:18,locat:8,log:[1,2,3,7,17],logg:0,logger:[5,19],loggeradapt:19,loglevel:[2,3,6,7],longer:3,look:[5,7],loop:[5,6],lorenzo:[0,7,8,24],low:[2,7],lower:[8,14],lst:[0,6,7,8,24],lx:[2,3],ly:[2,3],lz:3,m:[0,2,7,8,10,14],m_:25,m_c:2,m_phi:25,machado:0,made:3,magnitud:[3,8],mai:18,main:[2,7,10,14],mainli:[5,19],make:[2,3,5,6,7,8,14],manag:[1,2,3,12,15,17,20],mani:[],manipul:19,mansimd:[2,3],mansimdata:[1,2,3,6,7,8,17],manzanequ:0,map:[2,3,12],mar:0,mari:0,mario:0,martin:0,mat:[3,6,7,8],matemat:[],math:[0,1,2,3,7,8,9,14],mathemat:[0,5,9],matrix:7,matter:[6,7],max:[7,26],maximum:14,mcte:0,mean:[2,7],measur:[2,6,21],mechan:0,ment:5,mention:[2,5,7],merg:2,mesh:[3,6,12,15,18],mesh_dim:15,mesh_fil:3,mesh_wrapp:[],mesh_xdmf:3,meshwrapp:[],messag:[6,7,10,19,20],met:[12,15],meta:4,method:[0,2,5,6,7,10,12,14,15,17,18,19,20,24,25],metod:5,michael:0,micinn:0,might:[3,5,6,7,8,10,12,18],migth:[],min:7,min_tipcell_dist:15,minut:[2,3,5,6,7,8],mistak:[2,6],mix:[7,18],mixed_el:7,mixedel:7,mkdir:[2,3],moacf:10,mobil:25,mocaf:[1,4,5],mocafe_fold:[],mocef:[],model:[0,1,3,4,5,6,8,9,11,14,15,23,24,25],modifi:[2,19],modul:[7,9,11,12,15,18,19,23,24,25],moment:[],monitor:7,more:[2,5,7,10,15,18],moreov:2,most:[3,5,6,7,8],move:[0,2,3,15],move_tip_cel:[2,3,15],mpar:[2,3],mpi4pi:[18,22],mpi:[2,3,6,7,8,9,10,12,15,18,20,22],mpi_comm:[],mpirun:[2,3,6,7,8,10],msg:19,mshr:2,mu0:14,mu:[2,3,7,8,14],mu_0:[2,3],much:[],multidimension:[],multipl:20,multiple_pc_simul:6,must:[7,21,24,25,26],mv:[2,3],mx:0,my:[],my_sim_data:20,n:[0,2,3,5,6,7,8],n_chnk:18,n_global_mesh_vertic:[],n_local_mesh_vertic:[],n_point:13,n_sourc:[2,3,12],n_step:[2,3,6,7,8],n_variabl:18,nabla:[2,5,7,14,15],name:[2,7,18,20,21,24,25],name_param_a:21,name_param_b:21,nation:0,natur:[2,3,5,7,8],ncol:[2,3,6,7,8],ndarrai:[12,16],ndez:0,nearbi:6,necessari:[2,7],need:[2,3,5,6,7,8,10,18,19,20],nest:6,network:2,neumann:[2,3,5,7,8],never:5,new_posit:15,new_valu:21,newton:[],newtonl:[3,6,7,8],newtonsolv:22,next:[2,5,6,7,8],nl:22,non:[7,12],none:[2,3,6,7,8,12,20,22],nonlinear:[7,22],nonlinearproblem:22,norm:[2,15],normal:[3,8,18,19],notch:2,note:10,notebook:[2,3,4,5,6,7,8],noth:[5,7,12,15,19,20],notic:[2,3,5,6,7,8,10,18],now:[2,3,6,7,10],np:[6,7,8],number:[2,3,5,6,7,8,13,18],numer:[0,7],numpi:[6,7,8,10,12,16],nutrient:[2,7,24,25],nutrient_weak_form:[],nx:[2,3,6,7,8],ny:[2,3,6,7,8],nz:[3,8],object:[2,3,5,7,12,18,20,21,22,24,25],obscur:7,obtain:5,occur:[2,7,20],occurr:[],od:[2,3,10],odf:21,off:[2,3],offici:10,often:[2,5,6],old:5,oldid:0,omega:[2,5,14],onc:[2,3,6,7,8],one:[2,3,5,7,8,14,15,18,19,20],onli:[2,3,7,13,15,19],onlin:[0,5],onward:10,open:[5,9,10],openmpi:10,oper:[2,5,6,7],optim:7,option:[2,3,6,7,8,12,20],order:[2,3,5,6,7,8,14,18,20,26],ore:[24,25],org:[0,10],origin:[2,5,6,7,14,24,25],other:[2,5,6,7,10,12,15,21],otherwis:[2,12,15,18,20,21],our:[2,5,7],out:[7,10],outlin:7,output:[6,10],outsid:[5,7,16],outside_valu:[6,7,8,16],over:18,overrid:19,own:[7,19],oxygen:2,p0:[],p:[0,14,15,25],packag:[5,7,9,10],page:[2,3,6,7,8,9,10],panda:[10,21],paper:[2,6,7,14,24,25],parallel:[2,3,6,7,8,9,18,20],param:[18,25],param_df:21,param_dict:21,paramet:[1,2,3,6,7,8,9,12,13,14,15,17,18,19,20,24,25,26],parameters_fil:[2,3,18],paraview:[2,3,5,6,7,8],parent:[2,3,6,7,8],part:[2,12],partial:[0,2,5,7,14],particular:0,pass:[2,19],path:[2,3,6,7,8,10,19,20,21],pathlib:[2,3,6,7,8,19,20,21],pathwai:2,pattern:[0,7],pbar:[2,3],pc:6,pc_model:[6,7,8],pc_type:[3,6,7,8],pde:[0,2,5,7],peopl:5,perfectli:5,perform:[2,7],person:0,perspect:[],petsc4pi:[3,6,7,8],petsc:[3,6,7,8,22],petscmatrix:[3,6,7,8],petscnewtonsolv:22,petscproblem:[7,22],petscvector:[3,6,7,8],petter:0,phase:[0,1,2,3,4,6,7,8,9,11,12,14,15,16,23,24,25,26],phi0:[5,6,7,8],phi0_cpp_cod:[],phi0_in:[6,7,8],phi0_max:[],phi0_min:[],phi0_out:[6,7,8],phi:[5,6,7,8,24,25,26],phi_0:25,phi_c:15,phi_prec:24,phi_th:15,phi_xdmf:[5,6,7,8],philosophi:3,php:0,physic:5,pi:[2,15],pick:[2,7],pictur:7,piec:5,pip3:10,pip:9,place:[2,3,5,7,12,18,19],plan:[],platform:[4,5],pleas:[7,24,25],plo:0,ploson:0,pna:[0,7],png:[],point:[0,2,3,5,6,7,8,12,13,15,18],poir:0,polynomi:3,pone:0,popul:2,popular:5,portabl:7,posit:[2,3,12,15,18],possibl:[3,5,8,14,15],post:7,potenti:[14,24,25],pow:[3,5],power:[3,5,8],ppa:10,practic:[2,5],precis:[7,15,18],precondition:[],preconditioner_typ:3,prefer:[18,22],presenc:[5,7],present:[2,6,7,8,11,12,15,21,24],pretti:[7,25],previou:2,print:[6,7],problem:[0,3,6,7,8,22],proce:[2,3,10],procedur:[10,15],proceed:0,process:[2,3,6,7,8,12,15,18,19],product:25,progress:[2,3,6,7,8],progress_bar:[6,7,8],project:[0,2,3,5],prolif:25,prolifer:[2,7,14,15,24],properli:10,properti:10,prostat:[0,2,4,6,24],prostate_canc:[1,6,7,8,23],prostate_cancer2d:7,prostate_cancer3d:8,prostate_cancer_2d:7,prostate_cancer_3d:8,prostate_cancer_chem_potenti:24,prostate_cancer_form:[6,7,8,24],prostate_cancer_nutrient_form:[6,7,8,24],prostate_cancer_weak_form:[],provid:[2,3,4,5,6,7,9,10,12,15,18,20],pseudo:15,pt:0,publish:7,purpos:[2,7,9,10,15,18,19],put:[5,21],pvd:18,py:[2,3,5,6,7,8],pyhton:[],python3:[2,3,6,7,8,10],python:[0,2,3,4,5,6,7,8,9,10,16,19,21],python_fun:16,python_fun_param:16,pythonfunctionfield:16,q:14,quad:[2,5,14,15],quai:[],quit:2,r:[0,5,15,26],r_c:[2,15],r_v:3,radiu:[2,12,13,15,16],rand_max:[6,7,8],random:[6,7,8],random_sources_domain:2,randomli:[2,12,15],randomsourcemap:[2,3,12],randomst:[],rang:[2,3,5,6,7,8],rank:[2,3,6,7,8],rate:[2,7,14,15,24,25],rational:20,re:[2,3,5,7],reach:[],read:[2,3,5,7,14,15,26],reader:10,readi:[7,10],real:7,realli:[3,7,8],reason:[2,3,6,8],recommend:[2,3,5,6,7,8,10],rectangl:[2,5],rectanglemesh:[2,6,7],reduc:[2,7,8],refer:[2,7,21,24,25],refin:0,regard:[5,7],rel:[2,18],relat:[1,7,14],reload:3,remain:[2,3,8,14],remaind:18,remark:8,rememb:[2,3,8,24,25],remov:[2,12,15],remove_global_sourc:12,remove_sources_near_vessel:[2,3,12],repeat:6,report:[2,6,7,10,14,15,21,24],repositori:10,repres:[2,5,7,12,14,15,16,18,21,25,26],represent:5,reproduc:[2,7,9],requir:[2,3,5,7,8,18,24,25],resembl:[3,8],resolv:[2,3,6,7,8],respect:[2,7,12],respons:[2,12],result:[5,18,20],retriev:[2,7],revert:[2,3],revert_tip_cel:[2,3,15],revis:2,rf:10,richardson:0,right:[2,15],rim:25,ring:0,risk:2,rm:10,rmw:[],rodrguez:0,rogn:0,root:[2,19,20],rotational_expression_function_paramet:[],rotationalafexpressionfunct:[],round:6,routin:7,row:5,rui:0,rule:2,run:[5,10],run_prostate_cancer_simul:6,s:[0,2,3,5,6,7,8,24],s_:7,s_av:[6,7,8],s_averag:[6,7,8],s_exp:[6,7,8],s_express:[],s_max:[6,7,8],s_min:[6,7,8],salt:5,same:[2,3,6,7,8,11,14,18,20],save:[2,3,4,5,7,8,20],save_random_st:[],save_sim_info:[6,20],saw:6,scalar:14,scale:[0,7],scienc:0,scientif:[5,7,24,25],scott:0,script:[2,3,4,5,6,7,8,9,10],search:9,second:[2,3,5,6,7,8,14,20],section:[2,5,6,10],secur:[],see:[2,3,4,5,6,7,8,10,12,14,15],seen:2,select:[2,15],self:[2,7,22],semi:5,semiax:[7,16],semiax_i:[6,7,8,16],semiax_x:[6,7,8,16],semiax_z:[8,16],separ:[2,3,14,25],set:[2,3,5,6,7,8,16,19,20,21],set_descript:[3,6],set_equal_randomstate_for_all_p:[],set_log_level:[2,3,6,7],set_valu:[6,21],setfromopt:[3,6,7,8],setfunct:[3,6,7,8],setjacobian:[3,6,7,8],setstat:[],setup:6,setup_data_fold:[2,3,6,7,8,20],setup_pbar:3,setup_pvd_fil:18,setup_xdmf_fil:[2,3,6,7,8,18],sever:7,shape:6,share:[],sheet:[2,21],shell:10,shf:26,should:[5,10],show:[2,3,5,7,8],shown:2,shut:7,side:[2,3,5,7,8],sif:10,sigma0:[6,7,8],sigma0_in:[6,7,8],sigma0_out:[6,7,8],sigma:[6,7,8,24,25],sigma_0:25,sigma_old:24,sigma_xdmf:[6,7,8],sigmoid:[16,26],signal:2,significantli:7,sim_data:20,sim_descript:[6,20],sim_info:[6,20],sim_nam:[6,20],sim_rational:[],sim_valu:21,similar:3,similarli:[2,7],simparam:[2,3],simpi:[],simpl:[2,5,7,26],simplest:5,simpli:[2,3,5,6,7,8,10,14,20,25],simplier:[],simul:[0,1,4,5,9,11,12,14,15,17,18,20,21,25],sinc:[2,5,7,10,15],singl:[2,5],singular:9,singularityc:10,situat:5,slightli:[8,15],slope:[16,26],small:3,smooth:[16,26],smoothcircl:16,smoothcirculartumor:16,sne:[3,6,7,8],snes_solv:[3,6,7,8],snes_typ:[3,6,7,8],snesproblem:[3,6,7,8],so:[1,2,3,5,6,7,10,14,20],softw:0,softwar:[0,3,5,8,10],solut:[2,3,6,7,8],solv:[0,2,3,6,7,8,14],solvabl:[2,14],solver:[1,3,6,7,8,17],solver_paramet:22,solver_setup:22,solver_typ:3,some:[4,5,6,7,10,14,18],someth:[7,10],sometim:5,soon:2,sourc:[2,3,4,5,6,7,8,9,10,12],source_cel:12,source_map:12,source_point:12,sourcecel:12,sourcecellsmanag:2,sourcecomput:[],sourcemap:[2,12],sources_in_circle_point:12,sources_manag:[2,3],sources_map:[2,3],sourcesfield:[],sourcesmanag:[2,3,12],space:[2,5,7,8,18,26],spatial:[6,12,15],speak:[5,7],specif:[0,3,4,5,7,8,14,19],specifi:[5,7,20,24,25],sphere:[3,13,15],sphinx:[2,3,4,5,6,7,8],sphinx_gallery_thumbnail_path:[],spiral:[],spline:0,split:[2,3,6,7,8,14],spread:13,springer:0,sprout:2,squar:[2,5,7,18],stabl:[],standard:[2,3,6,7,14,19],standerd:19,start:[0,2,3,5,7],start_point:12,starv:2,state:[],statement:[2,3],std_out_all_process:[2,3],std_paramet:6,stem:2,step:[2,3,4,5,6,7,8,14,15,24,25],still:7,stop:2,store:[2,5,6,8,18,20,21],str:[18,19,20,21],string:[5,6,18],strong:0,structur:[5,21],studi:[],stuff:7,sub:[2,3,6,7,8],subclass:19,submodul:1,subpackag:[1,11,17,23],subspac:2,subsystem:10,sudo:10,suffici:2,suggest:[2,7],suit:[7,19],sum:[2,14],summar:5,summari:5,suppli:[7,24],support:10,suppos:7,sure:[2,3,6,7,8],surpass:12,surround:12,surviv:2,sy:7,symbol:5,sympli:[],system:[2,5,6,7,10],t:[0,2,3,5,6,7,8,10,14,25],t_c:15,t_valu:15,take:[2,3,6,7,8,10,12],tast:[],tau:[6,7,8,24],tc_p:15,team:14,techniqu:5,tell:[7,10],term:[2,5,7,14],termin:10,test:[2,3,6,10,14,24,25],testfunct:[2,3,5,6,7,8,14,24,25],tew:0,textrm:[2,5,14,15],than:[2,3,5,7,8,15,18],thei:[2,3,6,7,8,12],them:[2,3,6,7,8,9,10],theori:[],theta:[2,14],thi:[4,5,10,11,12,14,15,17,18,19,20,22,23,24,25],thing:[2,3,5,7,8],think:[],third:[2,5],thoma:0,thorugh:2,those:[5,7],though:[3,7,18],three:[2,6],threshold:[2,12],through:2,throughout:[2,7,15],thu:[2,3,6,8,15],time:[2,3,5,6,7,8,10,12,14,20,24,25],tip:[2,3,15],tip_cel:15,tip_cell_manag:[2,3],tip_cell_posit:15,tipcel:[1,2,3,11],tipcellmanag:[2,3,15],tipcells_field:[2,3],tipcells_xdmf:[2,3],tipcellsfield:15,tipcellsmanag:[2,3],tissu:[0,7,12,14,25],titl:0,todo:[],togeth:[2,5,6],told:[],too:[],took:2,tool:[6,15],toolkit:7,topic:5,total:[2,3,5,6,7,8,18],toward:7,tpoirec:[0,2,3,11,12,14,15],tqdm:[2,3,6,7,8,10],tradit:[],tran:0,transform:[],transit:10,translat:[5,7,12],travasso2011:[],travasso2011a:2,travasso:[0,2,3,11,12,14,15],tree:6,trial:[],trialfunct:5,triangl:7,trivial:6,tumor:[0,7,16,25],tumour:6,turn:[2,3],tutori:5,two:[2,3,6,7,14,15],type:[5,7,10,18,20],u:[2,3,5,6,7,8,25],u_0:5,u_0_exp:5,u_xdmf:5,ub:0,ubuntu:10,ufl:[2,5,7,14,24,25],um:[6,7,8],under:20,understand:[5,9],uni:0,unifi:[0,1,5],uniform:7,uninstal:9,uniqu:[2,6],unit:[2,21],unitsquaremesh:5,unknown:14,up:[2,3,5,6,7,8],updat:[2,3,5,6,7,8,10,15],uptak:[24,25],url:0,us:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,20,21,22,24,25,26],usabl:5,user:[2,5,7,16,19,20],usernam:[],usr:10,usual:[2,18],util:[1,17],v1:[2,3,6,7,8,14],v2:[2,3,6,7,8,14],v3:[2,3],v:[2,5,14,15,24,25,26],v_:25,v_pc:25,v_uh:25,v_ut:25,valu:[2,5,6,7,8,14,15,16,21,24,25,26],value_param_a:21,value_param_b:21,var_phi:24,variabl:[2,3,5,7,14,18,24,25,26],variant:5,variat:[3,7],varibl:26,varphi:[7,8,24,25,26],vascular:[0,2,12,14],vascular_proliferation_form:14,ve:8,vec:[3,6,7,8],vector:[2,3,6,7,8,15],vectorfunctionspac:[2,3],vegf:2,veloc:[2,15],velocity_norm:15,veri:[2,5,6,7],versatil:25,version:[0,3,5,8,10],vessel:[2,3,12,26],view:0,vilanova:0,visibl:[],visual:[5,18],vol:0,w:0,wa:[2,7,24],wai:[3,5,7,8,10,14,18,22],wander:[],wank:5,want:[2,5,6,7,10,18],warn:[],water:5,we:[2,3,5,6,7,8],weak:[0,1,3,6,8,14,24,25],weak_form:[2,3,6,7,8],websit:5,weight:14,well:[0,7,8,10,19,24,25],what:[2,6,7,8,10],when:[2,12,15,18,20],where:[2,3,4,5,7,12,14,15,18,19,26],which:[2,5,6,7,12,14,15,16,18,21,25,26],who:5,why:[],width:[2,18],wikipedia:0,wikipediacontributors21:[0,2,7],window:10,wise:5,wish:7,without:[2,7,20],won:10,wonder:[],word:5,work:[3,5,6,7,10,18,20,24,25,26],workflow:7,wrap:2,wrapper:[12,21],write:[2,3,5,6,7,8],written:5,wrong:7,wsl:10,www:[0,10],x:[2,3,5,7,15,26],x_max:[6,7,8],x_min:[6,7,8],xdmf:[2,3,5,6,7,8,18],xdmffile:5,xu16:[1,23],xu2016_nutrient_form:25,xu:[0,25],xu_2016_cancer_form:25,xvg16:[0,25],y:[6,10,26],y_max:[6,7,8],y_min:[6,7,8],year:[0,6,7,8],yongji:0,you:[1,2,3,4,5,6,7,8,9,10,18,19,20,24,25],your:[2,3,5,6,7,8,9,10,14,19,24,25],yourself:[5,7],z_max:8,z_min:8,zero:[2,7],zhang:0,zip:[4,6]},titles:["Bibliography","Code Documentation","Angiogenesis","Angiogenesis 3D","Demo Gallery","A brief introduction to FEniCS","Save simulations meta-data","Prostate cancer","Prostate cancer 3D","Welcome to <em>mocafe</em>\u2019s documentation!","Installation","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.af_sourcing</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.base_classes</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.forms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.angie.tipcells</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.expressions</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.fenut</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.log</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.mansimdata</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.parameters</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.fenut.solvers</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms.prostate_cancer</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.litforms.xu16</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">mocafe.math</span></code>"],titleterms:{"3d":[3,8],"function":3,A:5,In:[],The:[5,11,17,23],af_sourc:12,agent:2,angi:[11,12,13,14,15],angiogenesi:[2,3],apt:10,base:2,base_class:13,basic:5,below:[11,17,23],bibliographi:0,boundari:[2,3,5,7,8],brief:[2,5,7],can:[],cancer:[7,8],code:[1,2,3,5,6,7,8],complet:[],comput:2,condit:[2,3,5,7,8],contain:10,content:[],data:6,definit:[2,3,5,7,8],demo:4,differenti:2,diffus:5,discret:[2,5,7,8],document:[1,9,11,17,23],domain:3,each:[11,17,23],equat:[2,5],exampl:[2,3,6,7,8],express:16,fenic:[5,10],fenut:[17,18,19,20,21,22],field:[],find:[],follow:[],form:[2,5,7,14],full:[2,3,5,6,7,8,11,17,23],galleri:4,heat:5,how:[2,3,6,7,8],implement:[2,3,6,7,8],includ:[],index:[],indic:9,initi:[2,3,5,7,8],instal:10,introduct:[2,5,7],linux:[],litform:[23,24,25],log:19,manag:6,mansimdata:20,math:26,mathemat:[2,7],mesh:[2,5,7,8],meta:6,mocaf:[2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],model:[2,7],multipl:6,note:[3,8],packag:[],paramet:21,pde:[3,8],phase:[],pip:10,problem:5,prostat:[7,8],prostate_canc:24,provid:[11,17,23],recommend:[],remov:10,result:[2,3,6,7,8],run:[2,3,6,7,8],s:9,save:6,setup:[2,3,5,7,8],simul:[2,3,6,7,8],singular:10,solut:5,solv:5,solver:22,space:3,spatial:[2,3,5,7,8],submodul:[11,17,23],system:[3,8],tabl:9,thi:[2,3,6,7,8],tipcel:15,uninstal:10,visual:[2,3,6,7,8],weak:[2,5,7],welcom:9,what:5,workflow:5,xu16:25,you:[],your:[]}})