optimal_features = {'"hvac": {
				"k":2,
				"features": ['aggregate_7', 'seasonal_energy_8', 'max_trend_weekly', 'aggregate_5', 'seasonal_energy_7'],
				"accuracy":85
				},
			{
			"k":3,
			"features":['aggregate_7', 'aggregate_5', 'aggregate_3'],
			"accuracy":85
			}
			{
			"k":4,
			"features":['aggregate_7', 'aggregate_6', 'aggregate_11', 'gt_1000'],
			"accuracy":85
			},
			{
			k:5,
			"features":'aggregate_7', 'aggregate_11', 'aggregate_1', 'difference_min_max', 'aggregate_10', 'aggregate_6', 'seasonal_energy_6', 'seasonal_energy_9'],
			"accuracy":87,
			}
			,
			{
			k:1,
			"features": ['aggregate_5', 'aggregate_7', 'aggregate_4', 'aggregate_3', 'aggregate_1', 'fraction_17', 'seasonal_energy_5', 'autocorr', 'max_trend_daily', 'aggregate_2', 'aggregate_11'],
			"accuracy":84
			}
			},
		{
		"fridge":
			{
			"k":1,
			"features":['aggregate_1', 'fraction_11', 'aggregate_3', 'max_seasonal_12', 'aggregate_12', 'aggregate_4', 'aggregate_11', 'aggregate_2', 'total_occupants', 'fft_5', 'ratio_min_max'],
			"accuracy":88
			}
		}