import pandas as pd
from pandas import DataFrame
from typing import Dict,Any,Literal

from core.execution.dag_builder import DAGNode


class DAGExecutionError(Exception):
    pass

class UnsupportedOperationError(Exception):
    pass

class InvalidStrategyError(Exception):
    pass

class UnrecoverableRollbackError(Exception):
    pass

class DAGExecutor:
    def __init__(self, df, dag_builder, state_manager, logger):
        self.df            = df            
        self.dag_builder   = dag_builder
        self.state_manager = state_manager
        self.logger        = logger
        self.checkpoints   = {}            

    async def run(self,plan, mode, state_uuid) -> dict:
        dag = self.dag_builder.build(plan)
        self.logger.log(tool="DAG_EXECUTOR",
                        intent=f"Starting execution mode={mode}",
                        inputs={"node_count": len(dag.nodes),
                                "state_uuid": state_uuid},
                        outputs={}, confidence=1.0)
        actions_applied = []
        for node in dag.nodes:           
            try:
                self.checkpoints[node.id] = self.df.copy()
                self.df = await self._execute_node(node, mode)
                actions_applied.append({
                    "node_id"   : node.id,
                    "operation" : node.operation,
                    "column"    : node.column,
                    "status"    : "SUCCESS"
                })
                self.logger.log(tool="DAG_EXECUTOR",
                                intent=f"Node {node.id} executed",
                                inputs={"operation": node.operation},
                                outputs={"status": "SUCCESS"},
                                confidence=1.0)
            except Exception as e:
                self.logger.log(tool="DAG_EXECUTOR",
                                intent=f"Node {node.id} failed — rolling back",
                                inputs={"node_id": node.id},
                                outputs={"error": str(e)},
                                confidence=0.0)
                self._rollback(node.id)
                raise DAGExecutionError(node.id, node.operation, e)

        self.state_manager.save(state_uuid, dag, result={
            "rows_processed"  : len(self.df),
            "actions_applied" : actions_applied,
        })

        return {
            "df_transformed"  : self.df,
            "rows_processed"  : len(self.df),
            "actions_applied" : actions_applied,
            "dag_version"     : dag.version,
        }
    async def _execute_node(self,node: DAGNode, mode: str) -> DataFrame:
        target_df = self._get_target_df(self.df, mode)

        match node.operation:
            case "impute":
                return self._apply_impute(target_df, node)
            case "encode":
                return self._apply_encode(target_df, node)
            case "scale":
                return self._apply_scale(target_df, node)
            case "drop_column":
                return self._apply_drop(target_df, node)
            case "drop_duplicates":
                return self._apply_dedup(target_df, node)
            case "clip_outliers":
                return self._apply_clip(target_df, node)
            case "cast_dtype":
                return self._apply_cast(target_df, node)
            case "text_clean":
                return self._apply_text_clean(target_df, node)
            case _:
                raise UnsupportedOperationError(node.operation)
            
    def _apply_impute(self,df, node) -> DataFrame:
        strategy = node.params["strategy"]   # "mean" | "median" | "mode" | "constant"
        value    = node.params.get("fill_value")
        if strategy == "mean":
            df[node.column] = df[node.column].fillna(df[node.column].mean())
        elif strategy == "median":
            df[node.column] = df[node.column].fillna(df[node.column].median())
        elif strategy == "mode":
            df[node.column] = df[node.column].fillna(df[node.column].mode()[0])
        elif strategy == "constant":
            df[node.column] = df[node.column].fillna(value)
        else:
            raise InvalidStrategyError(strategy)
        return df
    
    def _apply_encode(self,df, node) -> DataFrame:
        method = node.params["method"]  
        if method == "ordinal":
            mapping = {v: i for i, v in enumerate(df[node.column].unique())}
            df[node.column] = df[node.column].map(mapping)
        elif method == "onehot":
            dummies = pd.get_dummies(df[node.column], prefix=node.column)
            df = pd.concat([df.drop(columns=[node.column]), dummies], axis=1)
        elif method == "target":
            target = node.params["target_column"]
            means  = df.groupby(node.column)[target].mean()
            df[node.column] = df[node.column].map(means)
        return df
    
    def _apply_scale(self,df, node) -> DataFrame:
        method = node.params["method"]   
        col = df[node.column]
        if method == "standard":
            df[node.column] = (col - col.mean()) / col.std()
        elif method == "minmax":
            df[node.column] = (col - col.min()) / (col.max() - col.min())
        elif method == "robust":
            q1, q3 = col.quantile(0.25), col.quantile(0.75)
            df[node.column] = (col - col.median()) / (q3 - q1)
        return df
    
    def _apply_drop(self,df, node) -> DataFrame:
        return df.drop(columns=[node.column], errors="ignore")
    
    def _apply_dedup(self,df, node) -> DataFrame:
        subset = node.params.get("subset")   
        return df.drop_duplicates(subset=subset)
    
    def _apply_clip(self,df, node) -> DataFrame:
        lower = node.params.get("lower")   # e.g. 1st percentile
        upper = node.params.get("upper")   # e.g. 99th percentile
        df[node.column] = df[node.column].clip(lower=lower, upper=upper)
        return df
    
    def _apply_cast(self,df, node) -> DataFrame:
        dtype = node.params["dtype"]   # "int" | "float" | "str" | "datetime"
        df[node.column] = df[node.column].astype(dtype)
        return df


    def _apply_text_clean(self,df, node) -> DataFrame:
        ops = node.params.get("ops", ["lowercase", "strip"])
        col = df[node.column].astype(str)

        if "lowercase" in ops:
            col = col.str.lower()
        if "strip" in ops:
            col = col.str.strip()
        if "remove_punct" in ops:
            col = col.str.replace(r"[^\w\s]", "", regex=True)

        df[node.column] = col
        return df


    def _rollback(self,failed_node_id):
        """
        Restore df to the snapshot taken just before
        the failed node executed.
        """
        if failed_node_id in self.checkpoints:
            self.df = self.checkpoints[failed_node_id]
            self.logger.log(tool="DAG_EXECUTOR",
                            intent=f"Rollback to pre-{failed_node_id} state",
                            inputs={}, outputs={"restored": True},
                            confidence=1.0)
        else:

            raise UnrecoverableRollbackError(failed_node_id)
        
    # async def write_outputs(self,result, job_id):
    #     output_path = build_output_path(job_id)
    #     DatasetWriter().write(
    #         df_transformed = result["df_transformed"],
    #         job_id         = job_id,
    #         output_dir     = output_path
    #     )
    #     PipelineExporter().export_and_save(
    #         dag         = result["dag_version"],
    #         output_path = output_path
    #     )

    def _get_target_df(df, mode) -> DataFrame:
        if mode == "DRY_RUN":
            return df.copy()       
        elif mode == "SAMPLE":
            return df.head(500).copy()
        else:
            return df     
