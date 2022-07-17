CREATE OR REPLACE FUNCTION
	weighted_fixed_array_inplace_update(
		csdw INTEGER[],
		csdc INTEGER[],
		cspv REAL[],
		cspc INTEGER[])
    RETURNS REAL[]
    SET SCHEMA 'public'
    LANGUAGE plpgsql
    AS $$
	DECLARE
		cpw REAL[];
		tw REAL[];
		tv REAL[];
		tc INTEGER;
    BEGIN
		-- Overall calculation
		-- result = (CSPV * CSPC + CSDW) / (CSPC + CSDC)

        -- TODO: Look at precision here

		-- cpw = cspv * cspc, tw = cpw + csdw, tc = cspc + csdc
		cpw = array_agg(e.el1 * e.el2) FROM unnest(cscv, cspv) e(el1, el2);
		tw = array_agg(e.el1 + e.el2) FROM unnest(cpw, csdw) e(el1, el2);
		tc = array_agg(e.el1 + e.el2) FROM unnest(cspc, csdc) e(el1, el2);

		-- tv = tw / tc
		tv = array_agg(e.el1 / e.el2) FROM unnest(tw, tc) e(el1, el2);

		RETURN tv;
	END;
$$;


CREATE OR REPLACE FUNCTION
	weighted_fixed_array_update(
		cscv REAL[],
		cscc INTEGER[],
		pscv REAL[],
		pscc INTEGER[],
		cspv REAL[],
		cspc INTEGER[])
    RETURNS REAL[]
    SET SCHEMA 'public'
    LANGUAGE plpgsql
    AS $$
	DECLARE
		ccw REAL[];
		pcw REAL[];
		ppw REAL[];
		twa REAL[];
		tw REAL[];
		tv REAL[];
		tc INTEGER[];
    BEGIN
		-- Overall calculation
		-- result = (CSCV * CSCC + PSCV * PSCC - CSPV * CSPC) / (CSCC + PSCC - CSPC)

        -- Need to cast to DOUBLE PRECISION and back to REAL otherwise we
        -- risk losing precision and even any effect if one term dominates.

		-- ccw = cscv * cscc, pcw = pscv * pscc, ppw = cspv * cspc
		ccw = array_agg(e.el1::DOUBLE PRECISION * e.el2) FROM unnest(cscv, cscc) e(el1, el2);
		pcw = array_agg(e.el1::DOUBLE PRECISION * e.el2) FROM unnest(pscv, pscc) e(el1, el2);
		ppw = array_agg(e.el1::DOUBLE PRECISION * e.el2) FROM unnest(cspv, cspc) e(el1, el2);

		-- twa = ccw + pcw, tw = twa - ppw
		twa = array_agg(e.el1 + e.el2) FROM unnest(ccw, pcw) e(el1, el2);
		tw = array_agg(e.el1 - e.el2) FROM unnest(twa, ppw) e(el1, el2);

        -- See fixed_array_update() which is the same as this line
		-- tc = cscc + pscc - cspc
        tc = array_agg(e.el1 + e.el2 - e.el3) FROM unnest(cscc, pscc, cspc) e(el1, el2, el3);

		-- tv = tw / tc
		tv = array_agg(e.el1 / e.el2)::REAL[] FROM unnest(tw, tc) e(el1, el2);

		RETURN tv;
	END;
$$;

CREATE OR REPLACE FUNCTION
	fixed_array_update(
		cscc INTEGER[],
		pscc INTEGER[],
		cspc INTEGER[])
    RETURNS REAL[]
    SET SCHEMA 'public'
    LANGUAGE plpgsql
    AS $$
    BEGIN
		-- Overall calculation
		-- result = CSCC + PSCC - CSPC
        RETURN array_agg(e.el1 + e.el2 - e.el3) FROM unnest(cscc, pscc, cspc) e(el1, el2, el3);
	END;
$$;

CREATE OR REPLACE FUNCTION
	fixed_vector_inplace_weights_update(
		csdc INTEGER[],
		cscc INTEGER[])
    RETURNS INTEGER[]
    SET SCHEMA 'public'
    LANGUAGE plpgsql
    AS $$
    BEGIN
		-- Overall calculation
		-- result = CSCC + CSDC
		RETURN array_agg(e.el1 + e.el2) FROM unnest(csdc, cscc) e(el1, el2);
	END;
$$;
