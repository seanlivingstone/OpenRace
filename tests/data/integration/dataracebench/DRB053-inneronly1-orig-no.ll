; ModuleID = 'integration/dataracebench/DRB053-inneronly1-orig-no.c'
source_filename = "integration/dataracebench/DRB053-inneronly1-orig-no.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@1 = private unnamed_addr constant [67 x i8] c";integration/dataracebench/DRB053-inneronly1-orig-no.c;main;60;1;;\00", align 1
@2 = private unnamed_addr constant [68 x i8] c";integration/dataracebench/DRB053-inneronly1-orig-no.c;main;60;25;;\00", align 1
@3 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #0 !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %a = alloca [20 x [20 x double]], align 16
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4, !tbaa !24
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !15, metadata !DIExpression()), !dbg !28
  store i8** %argv, i8*** %argv.addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !16, metadata !DIExpression()), !dbg !31
  %2 = bitcast i32* %i to i8*, !dbg !32
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #4, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %i, metadata !17, metadata !DIExpression()), !dbg !33
  %3 = bitcast i32* %j to i8*, !dbg !34
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %j, metadata !18, metadata !DIExpression()), !dbg !35
  %4 = bitcast [20 x [20 x double]]* %a to i8*, !dbg !36
  call void @llvm.lifetime.start.p0i8(i64 3200, i8* %4) #4, !dbg !36
  call void @llvm.dbg.declare(metadata [20 x [20 x double]]* %a, metadata !19, metadata !DIExpression()), !dbg !37
  %arraydecay = getelementptr inbounds [20 x [20 x double]], [20 x [20 x double]]* %a, i64 0, i64 0, !dbg !38
  %5 = bitcast [20 x double]* %arraydecay to i8*, !dbg !38
  call void @llvm.memset.p0i8.i64(i8* align 16 %5, i8 0, i64 3200, i1 false), !dbg !38
  store i32 0, i32* %i, align 4, !dbg !39, !tbaa !24
  br label %for.cond, !dbg !41

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !42, !tbaa !24
  %cmp = icmp slt i32 %6, 19, !dbg !44
  br i1 %cmp, label %for.body, label %for.end, !dbg !45

for.body:                                         ; preds = %for.cond
  %7 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !46
  store i8* getelementptr inbounds ([67 x i8], [67 x i8]* @1, i32 0, i32 0), i8** %7, align 8, !dbg !46, !tbaa !48
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* %.kmpc_loc.addr, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [20 x [20 x double]]*, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*), [20 x [20 x double]]* %a, i32* %i), !dbg !46
  br label %for.inc, !dbg !50

for.inc:                                          ; preds = %for.body
  %8 = load i32, i32* %i, align 4, !dbg !51, !tbaa !24
  %add = add nsw i32 %8, 1, !dbg !51
  store i32 %add, i32* %i, align 4, !dbg !51, !tbaa !24
  br label %for.cond, !dbg !52, !llvm.loop !53

for.end:                                          ; preds = %for.cond
  %9 = bitcast [20 x [20 x double]]* %a to i8*, !dbg !55
  call void @llvm.lifetime.end.p0i8(i64 3200, i8* %9) #4, !dbg !55
  %10 = bitcast i32* %j to i8*, !dbg !55
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %10) #4, !dbg !55
  %11 = bitcast i32* %i to i8*, !dbg !55
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %11) #4, !dbg !55
  ret i32 0, !dbg !56
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #2

; Function Attrs: norecurse nounwind uwtable
define internal void @.omp_outlined._debug__(i32* noalias %.global_tid., i32* noalias %.bound_tid., [20 x [20 x double]]* dereferenceable(3200) %a, i32* dereferenceable(4) %i) #3 !dbg !57 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %a.addr = alloca [20 x [20 x double]]*, align 8
  %i.addr = alloca i32*, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %j = alloca i32, align 4
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  store i32* %.global_tid., i32** %.global_tid..addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !67, metadata !DIExpression()), !dbg !77
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !68, metadata !DIExpression()), !dbg !77
  store [20 x [20 x double]]* %a, [20 x [20 x double]]** %a.addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata [20 x [20 x double]]** %a.addr, metadata !69, metadata !DIExpression()), !dbg !78
  store i32* %i, i32** %i.addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i32** %i.addr, metadata !70, metadata !DIExpression()), !dbg !79
  %2 = load [20 x [20 x double]]*, [20 x [20 x double]]** %a.addr, align 8, !dbg !80, !tbaa !29
  %3 = load i32*, i32** %i.addr, align 8, !dbg !80, !tbaa !29
  %4 = bitcast i32* %.omp.iv to i8*, !dbg !80
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %.omp.iv, metadata !71, metadata !DIExpression()), !dbg !77
  %5 = bitcast i32* %.omp.lb to i8*, !dbg !80
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %.omp.lb, metadata !72, metadata !DIExpression()), !dbg !77
  store i32 0, i32* %.omp.lb, align 4, !dbg !81, !tbaa !24
  %6 = bitcast i32* %.omp.ub to i8*, !dbg !80
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %.omp.ub, metadata !73, metadata !DIExpression()), !dbg !77
  store i32 19, i32* %.omp.ub, align 4, !dbg !81, !tbaa !24
  %7 = bitcast i32* %.omp.stride to i8*, !dbg !80
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %.omp.stride, metadata !74, metadata !DIExpression()), !dbg !77
  store i32 1, i32* %.omp.stride, align 4, !dbg !81, !tbaa !24
  %8 = bitcast i32* %.omp.is_last to i8*, !dbg !80
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %8) #4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %.omp.is_last, metadata !75, metadata !DIExpression()), !dbg !77
  store i32 0, i32* %.omp.is_last, align 4, !dbg !81, !tbaa !24
  %9 = bitcast i32* %j to i8*, !dbg !80
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #4, !dbg !80
  call void @llvm.dbg.declare(metadata i32* %j, metadata !76, metadata !DIExpression()), !dbg !77
  %10 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !80
  store i8* getelementptr inbounds ([67 x i8], [67 x i8]* @1, i32 0, i32 0), i8** %10, align 8, !dbg !80, !tbaa !48
  %11 = load i32*, i32** %.global_tid..addr, align 8, !dbg !80
  %12 = load i32, i32* %11, align 4, !dbg !80, !tbaa !24
  call void @__kmpc_for_static_init_4(%struct.ident_t* %.kmpc_loc.addr, i32 %12, i32 34, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1), !dbg !80
  %13 = load i32, i32* %.omp.ub, align 4, !dbg !81, !tbaa !24
  %cmp = icmp sgt i32 %13, 19, !dbg !81
  br i1 %cmp, label %cond.true, label %cond.false, !dbg !81

cond.true:                                        ; preds = %entry
  br label %cond.end, !dbg !81

cond.false:                                       ; preds = %entry
  %14 = load i32, i32* %.omp.ub, align 4, !dbg !81, !tbaa !24
  br label %cond.end, !dbg !81

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 19, %cond.true ], [ %14, %cond.false ], !dbg !81
  store i32 %cond, i32* %.omp.ub, align 4, !dbg !81, !tbaa !24
  %15 = load i32, i32* %.omp.lb, align 4, !dbg !81, !tbaa !24
  store i32 %15, i32* %.omp.iv, align 4, !dbg !81, !tbaa !24
  br label %omp.inner.for.cond, !dbg !80

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %cond.end
  %16 = load i32, i32* %.omp.iv, align 4, !dbg !81, !tbaa !24
  %17 = load i32, i32* %.omp.ub, align 4, !dbg !81, !tbaa !24
  %cmp1 = icmp sle i32 %16, %17, !dbg !80
  br i1 %cmp1, label %omp.inner.for.body, label %omp.inner.for.cond.cleanup, !dbg !80

omp.inner.for.cond.cleanup:                       ; preds = %omp.inner.for.cond
  br label %omp.inner.for.end, !dbg !80

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %18 = load i32, i32* %.omp.iv, align 4, !dbg !81, !tbaa !24
  %mul = mul nsw i32 %18, 1, !dbg !82
  %add = add nsw i32 0, %mul, !dbg !82
  store i32 %add, i32* %j, align 4, !dbg !82, !tbaa !24
  %19 = load i32, i32* %3, align 4, !dbg !83, !tbaa !24
  %add2 = add nsw i32 %19, 1, !dbg !85
  %idxprom = sext i32 %add2 to i64, !dbg !86
  %arrayidx = getelementptr inbounds [20 x [20 x double]], [20 x [20 x double]]* %2, i64 0, i64 %idxprom, !dbg !86
  %20 = load i32, i32* %j, align 4, !dbg !87, !tbaa !24
  %idxprom3 = sext i32 %20 to i64, !dbg !86
  %arrayidx4 = getelementptr inbounds [20 x double], [20 x double]* %arrayidx, i64 0, i64 %idxprom3, !dbg !86
  %21 = load double, double* %arrayidx4, align 8, !dbg !86, !tbaa !88
  %22 = load i32, i32* %3, align 4, !dbg !90, !tbaa !24
  %idxprom5 = sext i32 %22 to i64, !dbg !91
  %arrayidx6 = getelementptr inbounds [20 x [20 x double]], [20 x [20 x double]]* %2, i64 0, i64 %idxprom5, !dbg !91
  %23 = load i32, i32* %j, align 4, !dbg !92, !tbaa !24
  %idxprom7 = sext i32 %23 to i64, !dbg !91
  %arrayidx8 = getelementptr inbounds [20 x double], [20 x double]* %arrayidx6, i64 0, i64 %idxprom7, !dbg !91
  %24 = load double, double* %arrayidx8, align 8, !dbg !93, !tbaa !88
  %add9 = fadd double %24, %21, !dbg !93
  store double %add9, double* %arrayidx8, align 8, !dbg !93, !tbaa !88
  br label %omp.body.continue, !dbg !94

omp.body.continue:                                ; preds = %omp.inner.for.body
  br label %omp.inner.for.inc, !dbg !95

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %25 = load i32, i32* %.omp.iv, align 4, !dbg !81, !tbaa !24
  %add10 = add nsw i32 %25, 1, !dbg !80
  store i32 %add10, i32* %.omp.iv, align 4, !dbg !80, !tbaa !24
  br label %omp.inner.for.cond, !dbg !95, !llvm.loop !96

omp.inner.for.end:                                ; preds = %omp.inner.for.cond.cleanup
  br label %omp.loop.exit, !dbg !95

omp.loop.exit:                                    ; preds = %omp.inner.for.end
  %26 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !95
  store i8* getelementptr inbounds ([68 x i8], [68 x i8]* @2, i32 0, i32 0), i8** %26, align 8, !dbg !95, !tbaa !48
  call void @__kmpc_for_static_fini(%struct.ident_t* %.kmpc_loc.addr, i32 %12), !dbg !95
  %27 = bitcast i32* %j to i8*, !dbg !95
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %27) #4, !dbg !95
  %28 = bitcast i32* %.omp.is_last to i8*, !dbg !95
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %28) #4, !dbg !95
  %29 = bitcast i32* %.omp.stride to i8*, !dbg !95
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %29) #4, !dbg !95
  %30 = bitcast i32* %.omp.ub to i8*, !dbg !95
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %30) #4, !dbg !95
  %31 = bitcast i32* %.omp.lb to i8*, !dbg !95
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %31) #4, !dbg !95
  %32 = bitcast i32* %.omp.iv to i8*, !dbg !95
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %32) #4, !dbg !95
  ret void, !dbg !98
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

declare dso_local void @__kmpc_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32)

declare dso_local void @__kmpc_for_static_fini(%struct.ident_t*, i32)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: norecurse nounwind uwtable
define internal void @.omp_outlined.(i32* noalias %.global_tid., i32* noalias %.bound_tid., [20 x [20 x double]]* dereferenceable(3200) %a, i32* dereferenceable(4) %i) #3 !dbg !99 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %a.addr = alloca [20 x [20 x double]]*, align 8
  %i.addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !101, metadata !DIExpression()), !dbg !105
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !102, metadata !DIExpression()), !dbg !105
  store [20 x [20 x double]]* %a, [20 x [20 x double]]** %a.addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata [20 x [20 x double]]** %a.addr, metadata !103, metadata !DIExpression()), !dbg !105
  store i32* %i, i32** %i.addr, align 8, !tbaa !29
  call void @llvm.dbg.declare(metadata i32** %i.addr, metadata !104, metadata !DIExpression()), !dbg !105
  %0 = load [20 x [20 x double]]*, [20 x [20 x double]]** %a.addr, align 8, !dbg !106, !tbaa !29
  %1 = load i32*, i32** %i.addr, align 8, !dbg !106, !tbaa !29
  %2 = load i32*, i32** %.global_tid..addr, align 8, !dbg !106, !tbaa !29
  %3 = load i32*, i32** %.bound_tid..addr, align 8, !dbg !106, !tbaa !29
  %4 = load [20 x [20 x double]]*, [20 x [20 x double]]** %a.addr, align 8, !dbg !106, !tbaa !29
  %5 = load i32*, i32** %i.addr, align 8, !dbg !106, !tbaa !29
  call void @.omp_outlined._debug__(i32* %2, i32* %3, [20 x [20 x double]]* %4, i32* %5) #4, !dbg !106
  ret void, !dbg !106
}

declare !callback !107 dso_local void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.1 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "integration/dataracebench/DRB053-inneronly1-orig-no.c", directory: "/home/brad/Code/OpenRace/tests/data")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.1 "}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 52, type: !8, scopeLine: 53, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !{!15, !16, !17, !18, !19}
!15 = !DILocalVariable(name: "argc", arg: 1, scope: !7, file: !1, line: 52, type: !10)
!16 = !DILocalVariable(name: "argv", arg: 2, scope: !7, file: !1, line: 52, type: !11)
!17 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 54, type: !10)
!18 = !DILocalVariable(name: "j", scope: !7, file: !1, line: 55, type: !10)
!19 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 56, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 25600, elements: !22)
!21 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!22 = !{!23, !23}
!23 = !DISubrange(count: 20)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DILocation(line: 52, column: 14, scope: !7)
!29 = !{!30, !30, i64 0}
!30 = !{!"any pointer", !26, i64 0}
!31 = !DILocation(line: 52, column: 25, scope: !7)
!32 = !DILocation(line: 54, column: 3, scope: !7)
!33 = !DILocation(line: 54, column: 7, scope: !7)
!34 = !DILocation(line: 55, column: 3, scope: !7)
!35 = !DILocation(line: 55, column: 7, scope: !7)
!36 = !DILocation(line: 56, column: 3, scope: !7)
!37 = !DILocation(line: 56, column: 10, scope: !7)
!38 = !DILocation(line: 57, column: 3, scope: !7)
!39 = !DILocation(line: 59, column: 10, scope: !40)
!40 = distinct !DILexicalBlock(scope: !7, file: !1, line: 59, column: 3)
!41 = !DILocation(line: 59, column: 8, scope: !40)
!42 = !DILocation(line: 59, column: 15, scope: !43)
!43 = distinct !DILexicalBlock(scope: !40, file: !1, line: 59, column: 3)
!44 = !DILocation(line: 59, column: 17, scope: !43)
!45 = !DILocation(line: 59, column: 3, scope: !40)
!46 = !DILocation(line: 60, column: 1, scope: !47)
!47 = distinct !DILexicalBlock(scope: !43, file: !1, line: 59, column: 34)
!48 = !{!49, !30, i64 16}
!49 = !{!"ident_t", !25, i64 0, !25, i64 4, !25, i64 8, !25, i64 12, !30, i64 16}
!50 = !DILocation(line: 64, column: 3, scope: !47)
!51 = !DILocation(line: 59, column: 28, scope: !43)
!52 = !DILocation(line: 59, column: 3, scope: !43)
!53 = distinct !{!53, !45, !54}
!54 = !DILocation(line: 64, column: 3, scope: !40)
!55 = !DILocation(line: 66, column: 1, scope: !7)
!56 = !DILocation(line: 65, column: 3, scope: !7)
!57 = distinct !DISubprogram(name: ".omp_outlined._debug__", scope: !1, file: !1, line: 61, type: !58, scopeLine: 61, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !66)
!58 = !DISubroutineType(types: !59)
!59 = !{null, !60, !60, !64, !65}
!60 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !61)
!61 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !62)
!62 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !63, size: 64)
!63 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!64 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !20, size: 64)
!65 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !10, size: 64)
!66 = !{!67, !68, !69, !70, !71, !72, !73, !74, !75, !76}
!67 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !57, type: !60, flags: DIFlagArtificial)
!68 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !57, type: !60, flags: DIFlagArtificial)
!69 = !DILocalVariable(name: "a", arg: 3, scope: !57, file: !1, line: 56, type: !64)
!70 = !DILocalVariable(name: "i", arg: 4, scope: !57, file: !1, line: 54, type: !65)
!71 = !DILocalVariable(name: ".omp.iv", scope: !57, type: !10, flags: DIFlagArtificial)
!72 = !DILocalVariable(name: ".omp.lb", scope: !57, type: !10, flags: DIFlagArtificial)
!73 = !DILocalVariable(name: ".omp.ub", scope: !57, type: !10, flags: DIFlagArtificial)
!74 = !DILocalVariable(name: ".omp.stride", scope: !57, type: !10, flags: DIFlagArtificial)
!75 = !DILocalVariable(name: ".omp.is_last", scope: !57, type: !10, flags: DIFlagArtificial)
!76 = !DILocalVariable(name: "j", scope: !57, type: !10, flags: DIFlagArtificial)
!77 = !DILocation(line: 0, scope: !57)
!78 = !DILocation(line: 56, column: 10, scope: !57)
!79 = !DILocation(line: 54, column: 7, scope: !57)
!80 = !DILocation(line: 61, column: 5, scope: !57)
!81 = !DILocation(line: 61, column: 10, scope: !57)
!82 = !DILocation(line: 61, column: 25, scope: !57)
!83 = !DILocation(line: 62, column: 20, scope: !84)
!84 = distinct !DILexicalBlock(scope: !57, file: !1, line: 61, column: 33)
!85 = !DILocation(line: 62, column: 22, scope: !84)
!86 = !DILocation(line: 62, column: 18, scope: !84)
!87 = !DILocation(line: 62, column: 27, scope: !84)
!88 = !{!89, !89, i64 0}
!89 = !{!"double", !26, i64 0}
!90 = !DILocation(line: 62, column: 9, scope: !84)
!91 = !DILocation(line: 62, column: 7, scope: !84)
!92 = !DILocation(line: 62, column: 12, scope: !84)
!93 = !DILocation(line: 62, column: 15, scope: !84)
!94 = !DILocation(line: 63, column: 5, scope: !84)
!95 = !DILocation(line: 60, column: 1, scope: !57)
!96 = distinct !{!96, !95, !97}
!97 = !DILocation(line: 60, column: 25, scope: !57)
!98 = !DILocation(line: 63, column: 5, scope: !57)
!99 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 61, type: !58, scopeLine: 61, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !100)
!100 = !{!101, !102, !103, !104}
!101 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !99, type: !60, flags: DIFlagArtificial)
!102 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !99, type: !60, flags: DIFlagArtificial)
!103 = !DILocalVariable(name: "a", arg: 3, scope: !99, type: !64, flags: DIFlagArtificial)
!104 = !DILocalVariable(name: "i", arg: 4, scope: !99, type: !65, flags: DIFlagArtificial)
!105 = !DILocation(line: 0, scope: !99)
!106 = !DILocation(line: 61, column: 5, scope: !99)
!107 = !{!108}
!108 = !{i64 2, i64 -1, i64 -1, i1 true}
