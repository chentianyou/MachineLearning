package main

import (
	"encoding/json"
	"log"
	"fmt"
)

const (
	IsFeature = iota
	IsLabel
	Other
)

type Config struct {
	DataDir    string      `json:"data_dir"`
	ModelDir   string      `json:"model_dir"`
	NCluster   int64       `json:"n_cluster"`
	Epoch      int         `json:"epoch"`
	BatchSize  int         `json:"batch_size"`
	NumInput   int         `json:"num_input"`
	DataConfig *DataConfig `json:"data_config"`
}

type DataConfig struct {
	Columns     []Column      `json:"columns"`
	CrossColumn []CrossColumn `json:"cross_column"`
}

type CrossColumn struct {
	Columns    []string             `json:"columns"`
	Boundaries map[string][]float32 `json:"boundaries"`
}

type Column struct {
	Name   string   `json:"name"`
	Type   string   `json:"type"`
	Index  int64    `json:"index"`
	Kind   int      `json:"kind"`
	NClass *int64   `json:"n_class"`
	Mean   *float64 `json:"mean"`
	StdDev *float64 `json:"std_dev"`
}

func main() {
	config := Config{}
	config.DataDir = "data/"
	config.ModelDir = "model_dir/iris"
	config.NCluster = 3
	config.Epoch = 10
	config.BatchSize = 15
	config.NumInput = 145
	dataConfig := DataConfig{}
	mean := []float64{
		5.84620689655172,
		3.04275862068966,
		3.79517241379311,
		1.21586206896552,
	}
	stdDev := []float64{
		0.826126874254926,
		0.433708121712589,
		1.75382659601631,
		0.759371415377397,
	}
	dataConfig.Columns = append(dataConfig.Columns, Column{
		Name:   "key1",
		Type:   "DT_FLOAT",
		Index:  0,
		Kind:   IsFeature,
		NClass: nil,
		Mean:   &mean[0],
		StdDev: &stdDev[0],
	})
	dataConfig.Columns = append(dataConfig.Columns, Column{
		Name:   "key2",
		Type:   "DT_FLOAT",
		Index:  1,
		Kind:   IsFeature,
		NClass: nil,
		Mean:   &mean[1],
		StdDev: &stdDev[1],
	})
	dataConfig.Columns = append(dataConfig.Columns, Column{
		Name:   "key3",
		Type:   "DT_FLOAT",
		Index:  2,
		Kind:   IsFeature,
		NClass: nil,
		Mean:   &mean[2],
		StdDev: &stdDev[2],
	})
	dataConfig.Columns = append(dataConfig.Columns, Column{
		Name:   "key4",
		Type:   "DT_FLOAT",
		Index:  3,
		Kind:   IsFeature,
		NClass: nil,
		Mean:   &mean[3],
		StdDev: &stdDev[3],
	})
	dataConfig.Columns = append(dataConfig.Columns, Column{
		Name:   "key5",
		Type:   "DT_STRING",
		Index:  4,
		Kind:   IsLabel,
		NClass: nil,
		Mean:   nil,
		StdDev: nil,
	})
	boundaries := make(map[string][]float32)
	boundaries["key1"] = []float32{5, 6}
	boundaries["key2"] = []float32{3, 4}
	dataConfig.CrossColumn = append(dataConfig.CrossColumn, CrossColumn{
		Columns:    []string{"key1", "key2"},
		Boundaries: boundaries,
	})
	config.DataConfig = &dataConfig
	buff, err := json.Marshal(&config)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(buff))
}
