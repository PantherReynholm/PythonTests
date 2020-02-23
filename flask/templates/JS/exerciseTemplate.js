class Excercises {
    constructor(weight, reps, sets) {
        this.weight = weight;
        this.reps = reps;
        this.sets = sets;
    }

    burned_calories() {
        let calories = 0;
        let counter = 0;
        this.sets.array.forEach(element => {
            calories += element * this.reps[counter] * this.weight;
            counter++ 
        });
        return calories
    }
}


class AlternateDumbellBenchPress extends Excercises {
    constructor() {
        super(Excercises);
        this.height = 0.45
    }
}
